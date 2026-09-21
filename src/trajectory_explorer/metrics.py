"""Per-tensor delta metrics: norms, spectrum, effective rank, row concentration.

All reductions accumulate in float64 over bounded chunks, so memory stays at roughly the size
of the two input tensors plus their difference, even for a 50304 x 512 embedding.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from trajectory_explorer import noise

# Bump when any metric's meaning changes; cached results with another version are recomputed.
# v2: raw measurements record whether F32 tensors hold only float16-exact values (D40).
METRICS_VERSION = 2

# float64 elements per chunk (4M elements = 32 MB).
DEFAULT_CHUNK_ELEMENTS = 4_000_000
ENERGY_FRACTION = 0.9  # r90: number of singular values holding 90% of the delta's energy


def _f16_exact(chunk: np.ndarray) -> bool:
    """True if every float32 value survives float32 -> float16 -> float32 bit for bit.

    NaN counts as representable when it round-trips to NaN (payload bits may change); +-inf
    round-trips exactly; finite values beyond float16's range become inf and fail.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        back = chunk.astype(np.float16).astype(np.float32)
    same = chunk.view(np.uint32) == back.view(np.uint32)
    if same.all():
        return True
    return bool((same | (np.isnan(chunk) & np.isnan(back))).all())


def sum_squares_and_f16(
    x: np.ndarray, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS, *, check_f16: bool = False
) -> tuple[float, bool | None]:
    """Sum of squares (float64 accumulation) and, optionally, float16 exactness, in one pass.

    Works on bounded chunks, so no full extra copy of a big tensor is made.
    """
    flat = x.reshape(-1)
    total = 0.0
    exact: bool | None = True if check_f16 else None
    for start in range(0, flat.size, chunk_elements):
        block = flat[start : start + chunk_elements]
        if exact:
            exact = _f16_exact(np.ascontiguousarray(block, dtype=np.float32))
        chunk = block.astype(np.float64)
        total += float(np.dot(chunk, chunk))
    return total, exact


def sum_squares(x: np.ndarray, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> float:
    """Sum of squares with float64 accumulation, one bounded chunk at a time."""
    return sum_squares_and_f16(x, chunk_elements)[0]


def gram_singular_values(d: np.ndarray, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> np.ndarray:
    """Singular values of a 2-D matrix, descending, via the smaller Gram matrix.

    For an m x n matrix with n <= m this builds the n x n matrix D^T D (else D D^T) in float64,
    accumulated over row (or column) chunks, and takes its eigenvalues: sigma_i = sqrt(lambda_i).
    Squaring loses singular values below ~1e-8 of the largest; they carry no weight in the
    entropy rank or r90.
    """
    rows, cols = d.shape
    if cols <= rows:
        gram = np.zeros((cols, cols), dtype=np.float64)
        step = max(1, chunk_elements // max(cols, 1))
        for start in range(0, rows, step):
            block = d[start : start + step].astype(np.float64)
            gram += block.T @ block
    else:
        gram = np.zeros((rows, rows), dtype=np.float64)
        step = max(1, chunk_elements // max(rows, 1))
        for start in range(0, cols, step):
            block = d[:, start : start + step].astype(np.float64)
            gram += block @ block.T
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)[::-1]
    return np.sqrt(eigenvalues)


def entropy_effective_rank(singular_values: np.ndarray) -> float | None:
    """Roy & Vetterli (2007): exp(Shannon entropy of sigma / sum(sigma)). None if all zero."""
    total = float(singular_values.sum())
    if total <= 0.0:
        return None
    p = singular_values[singular_values > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def energy_rank(singular_values: np.ndarray, fraction: float = ENERGY_FRACTION) -> int | None:
    """Smallest k such that the top-k singular values hold ``fraction`` of sum(sigma^2)."""
    energy = singular_values.astype(np.float64) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return None
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, fraction - 1e-12)) + 1


@lru_cache(maxsize=256)
def random_erank_ratio(n_small: int, n_large: int, points: int = 4096) -> float:
    """Expected erank / n_small for an n_small x n_large Gaussian matrix (Marchenko-Pastur).

    With aspect ratio lam = n_small / n_large, squared singular values (scaled) follow the
    Marchenko-Pastur law on [(1 - sqrt(lam))^2, (1 + sqrt(lam))^2]. For singular values s with
    mean mu, erank / n = mu * exp(-E[s log s] / mu). The integral uses a cosine substitution
    that removes the square-root edges, then the midpoint rule.
    """
    if n_small <= 0 or n_large <= 0:
        raise ValueError("matrix dimensions must be positive")
    lam = min(n_small, n_large) / max(n_small, n_large)
    lo, hi = (1 - math.sqrt(lam)) ** 2, (1 + math.sqrt(lam)) ** 2
    theta = (np.arange(points) + 0.5) * (math.pi / points)
    x = lo + (hi - lo) * (1 - np.cos(theta)) / 2
    weights = np.sin(theta) ** 2 / x  # density * dx, up to a constant
    weights /= weights.sum()
    s = np.sqrt(x)
    mu = float(np.sum(weights * s))
    return float(mu * math.exp(-float(np.sum(weights * s * np.log(s))) / mu))


def top_row_share(
    d: np.ndarray,
    fraction: float = noise.TOP_ROW_FRACTION,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> float | None:
    """Share of the delta's squared norm held by the top ``fraction`` of rows (axis 0).

    For a 1-D tensor each element is a row. None for scalars or an all-zero delta.
    """
    if d.ndim == 0 or d.shape[0] == 0:
        return None
    rows = d.reshape(d.shape[0], -1)
    step = max(1, chunk_elements // max(rows.shape[1], 1))
    row_energy = np.empty(rows.shape[0], dtype=np.float64)
    for start in range(0, rows.shape[0], step):
        block = rows[start : start + step].astype(np.float64)
        row_energy[start : start + step] = np.einsum("ij,ij->i", block, block)
    total = float(row_energy.sum())
    if total <= 0.0:
        return None
    k = max(1, math.ceil(fraction * row_energy.size))
    top = np.partition(row_energy, row_energy.size - k)[-k:]
    return float(top.sum() / total)


@dataclass(frozen=True)
class TensorMeasurement:
    """Raw measurements of one tensor's delta B - A, independent of floor and control.

    This is what the metrics cache stores; ``assess_tensor`` derives everything else.
    """

    shape: tuple[int, ...]
    dtype_a: str  # stored dtypes: they set the rounding floor
    dtype_b: str
    norm_a: float
    norm_b: float
    abs_delta: float  # ||B - A||_F
    erank: float | None  # entropy effective rank of B - A (matrices only)
    erank_random: float | None  # same for a Gaussian matrix of this shape
    r90: int | None
    rank_dims: int | None  # min(rows, cols): the largest possible rank
    top_row_share: float | None
    # Every value is exactly a float16 number: True for F16 storage, checked for F32 storage,
    # None (not applicable) for other stored dtypes.
    exact_f16_a: bool | None = None
    exact_f16_b: bool | None = None

    @property
    def rel_delta(self) -> float | None:
        """||B - A|| / ||A||; 0.0 if both are zero; None if A is zero and B is not."""
        if self.norm_a > 0.0:
            return self.abs_delta / self.norm_a
        return 0.0 if self.abs_delta == 0.0 else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMeasurement:
        return cls(**{**data, "shape": tuple(data["shape"])})


@dataclass(frozen=True)
class TensorMetrics:
    """A measurement plus its assessment (floor, control, status, labels). JSON-safe."""

    shape: tuple[int, ...]
    dtype_a: str
    dtype_b: str
    norm_a: float
    norm_b: float
    abs_delta: float  # ||B - A||_F
    rel_delta: float | None  # ||B - A||_F / ||A||_F; None when ||A|| == 0
    from_zero: bool  # A is exactly zero and B is not
    erank: float | None
    erank_random: float | None
    r90: int | None
    rank_dims: int | None
    top_row_share: float | None
    floor: float  # rounding floor on rel_delta (from the effective precision, D40)
    control: float | None  # optional reference scale on rel_delta
    status: str  # see noise.assess
    rank_label: str  # "low-rank" | "dense" | "n/a"
    concentration_label: str  # "concentrated" | "spread" | "n/a"
    # True when a float32-stored side was treated as float16 for the floor (D40).
    f16_rule: bool = False
    exact_f16_a: bool | None = None  # copied from the measurement
    exact_f16_b: bool | None = None

    @property
    def significant(self) -> bool:
        return self.status in noise.SIGNIFICANT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMetrics:
        return cls(**{**data, "shape": tuple(data["shape"])})


def measure_tensor(
    a: np.ndarray,
    b: np.ndarray,
    dtype_a: str,
    dtype_b: str,
    *,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> TensorMeasurement:
    """Measure the delta between two float32 tensors of the same shape.

    Raises ValueError for mismatched shapes or non-finite values.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = np.subtract(b, a, dtype=np.float32)

    ss_a, exact_a = sum_squares_and_f16(a, chunk_elements, check_f16=dtype_a == "F32")
    ss_b, exact_b = sum_squares_and_f16(b, chunk_elements, check_f16=dtype_b == "F32")
    exact_a = True if dtype_a == "F16" else exact_a
    exact_b = True if dtype_b == "F16" else exact_b
    ss_d = sum_squares(d, chunk_elements)
    if not all(math.isfinite(v) for v in (ss_a, ss_b, ss_d)):
        raise ValueError("tensor contains NaN or inf values")
    abs_delta = math.sqrt(ss_d)

    erank = erank_random = None
    r90 = rank_dims = None
    if d.ndim >= 2 and abs_delta > 0.0:
        matrix = d.reshape(d.shape[0], -1)
        rows, cols = matrix.shape
        if min(rows, cols) > 1:
            sv = gram_singular_values(matrix, chunk_elements)
            erank = entropy_effective_rank(sv)
            r90 = energy_rank(sv)
            rank_dims = min(rows, cols)
            erank_random = random_erank_ratio(rank_dims, max(rows, cols)) * rank_dims

    return TensorMeasurement(
        shape=tuple(int(s) for s in a.shape),
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        norm_a=math.sqrt(ss_a),
        norm_b=math.sqrt(ss_b),
        abs_delta=abs_delta,
        erank=erank,
        erank_random=erank_random,
        r90=r90,
        rank_dims=rank_dims,
        top_row_share=top_row_share(d, chunk_elements=chunk_elements) if abs_delta else None,
        exact_f16_a=exact_a,
        exact_f16_b=exact_b,
    )


def effective_dtypes(m: TensorMeasurement) -> tuple[str, str, bool]:
    """Stored dtypes, except float32 sides count as float16 when both sides hold only
    float16-exact values (D40). Returns (dtype_a, dtype_b, rule_applied)."""
    both_f16 = m.exact_f16_a is True and m.exact_f16_b is True
    if not both_f16 or "F32" not in (m.dtype_a, m.dtype_b):
        return m.dtype_a, m.dtype_b, False
    return (
        "F16" if m.dtype_a == "F32" else m.dtype_a,
        "F16" if m.dtype_b == "F32" else m.dtype_b,
        True,
    )


def assess_tensor(m: TensorMeasurement, control: float | None = None) -> TensorMetrics:
    """Apply the rounding floor, the optional control scale and the labels to a measurement."""
    rel_delta = m.rel_delta
    eff_a, eff_b, f16_rule = effective_dtypes(m)
    floor = noise.dtype_floor(eff_a, eff_b)
    return TensorMetrics(
        shape=m.shape,
        dtype_a=m.dtype_a,
        dtype_b=m.dtype_b,
        norm_a=m.norm_a,
        norm_b=m.norm_b,
        abs_delta=m.abs_delta,
        rel_delta=rel_delta,
        from_zero=m.norm_a == 0.0 and m.abs_delta > 0.0,
        erank=m.erank,
        erank_random=m.erank_random,
        r90=m.r90,
        rank_dims=m.rank_dims,
        top_row_share=m.top_row_share,
        floor=floor,
        control=control,
        status=noise.assess(rel_delta, m.abs_delta, floor, control),
        rank_label=noise.rank_label(m.erank, m.erank_random),
        # Matrices only (D43): for vectors the top-5% share is shown as a number instead.
        concentration_label=(
            noise.concentration_label(m.top_row_share, m.shape[0]) if len(m.shape) >= 2 else "n/a"
        ),
        f16_rule=f16_rule,
        exact_f16_a=m.exact_f16_a,
        exact_f16_b=m.exact_f16_b,
    )


def compute_tensor_metrics(
    a: np.ndarray,
    b: np.ndarray,
    dtype_a: str,
    dtype_b: str,
    *,
    control: float | None = None,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> TensorMetrics:
    """Measure and assess in one step (``measure_tensor`` + ``assess_tensor``)."""
    measurement = measure_tensor(a, b, dtype_a, dtype_b, chunk_elements=chunk_elements)
    return assess_tensor(measurement, control)

"""Diff engine: two checkpoint files -> DiffResult (per tensor + per (layer, component) group).

Flow: content hashes (sha256 of local files, or the Hub's LFS sha256) -> byte-identical
short-circuit -> raw measurements from the JSON cache or computed one tensor at a time ->
floor/control/labels applied -> groups. Files are fetched only when a pair must actually be
measured, and the compared pair is finished (and cached) before the control pair is fetched,
so a two-file rolling store never has to re-read an evicted checkpoint.

Only raw measurements are cached (keyed by sha256(A), sha256(B), METRICS_VERSION). The floor,
the control scale and the labels are applied after loading, so a different --control never
invalidates the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from trajectory_explorer import metrics, noise
from trajectory_explorer._version import __version__
from trajectory_explorer.arch import check_compatible, classify, parameter_specs
from trajectory_explorer.errors import ArchitectureMismatch, CheckpointFormatError, InputError
from trajectory_explorer.metrics import TensorMeasurement, TensorMetrics, assess_tensor
from trajectory_explorer.reader import Checkpoint

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CACHE_FORMAT = 1
_HASH_CHUNK = 4 * 1024 * 1024


# -- data model ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceInfo:
    label: str  # short name for titles, e.g. "step2000" or "pythia-70m@step2000"
    path: str  # local path, or "org/name@revision" for Hub sources
    sha256: str
    size_bytes: int


class CheckpointSource(Protocol):
    """Anything with known content identity that can produce a local file on demand."""

    info: SourceInfo

    def fetch(self) -> AbstractContextManager[Path]:
        """Yield a local path to the checkpoint, valid (and pinned) inside the with-block."""
        ...


@dataclass(frozen=True)
class LocalSource:
    path: Path
    info: SourceInfo

    def fetch(self) -> AbstractContextManager[Path]:
        return nullcontext(self.path)


@dataclass(frozen=True)
class GroupMetrics:
    """Aggregate over the tensors of one (layer, component) cell.

    rel_delta = sqrt(sum ||dW||^2) / sqrt(sum ||W_A||^2). The group floor is the rounding
    noise the members would show together: sqrt(sum (floor_i * ||W_A,i||)^2) / sqrt(sum
    ||W_A,i||^2). If every member is at or below its own floor, the group is too.
    """

    layer: int | None
    component: str
    tensors: tuple[str, ...]
    norm_a: float | None  # None when the files are byte-identical (nothing was measured)
    abs_delta: float
    rel_delta: float | None  # None when the group's reference norm is exactly zero
    floor: float
    control: float | None
    status: str

    @property
    def significant(self) -> bool:
        return self.status in noise.SIGNIFICANT_STATUSES


@dataclass(frozen=True)
class DiffResult:
    a: SourceInfo
    b: SourceInfo
    identical: bool  # byte-identical files: nothing was measured
    n_tensors: int
    skipped: tuple[str, ...]  # buffers and non-float tensors, never compared
    tensors: dict[str, TensorMetrics]
    groups: tuple[GroupMetrics, ...]
    control: tuple[SourceInfo, SourceInfo] | None = None
    schema_version: int = SCHEMA_VERSION
    metrics_version: int = field(default_factory=lambda: metrics.METRICS_VERSION)
    tool_version: str = __version__

    @property
    def significant_tensors(self) -> list[str]:
        return [name for name, m in self.tensors.items() if m.significant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics_version": self.metrics_version,
            "tool_version": self.tool_version,
            "a": asdict(self.a),
            "b": asdict(self.b),
            "control": [asdict(s) for s in self.control] if self.control else None,
            "identical": self.identical,
            "n_tensors": self.n_tensors,
            "skipped": list(self.skipped),
            "tensors": {name: m.to_dict() for name, m in self.tensors.items()},
            "groups": [{**asdict(g), "tensors": list(g.tensors)} for g in self.groups],
        }

    def to_json(self) -> str:
        """Serialize; NaN or inf anywhere raises ValueError instead of writing invalid JSON."""
        return json.dumps(self.to_dict(), allow_nan=False, indent=1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffResult:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported DiffResult schema_version {data.get('schema_version')}")
        control = data.get("control")
        return cls(
            a=SourceInfo(**data["a"]),
            b=SourceInfo(**data["b"]),
            identical=data["identical"],
            n_tensors=data["n_tensors"],
            skipped=tuple(data["skipped"]),
            tensors={n: TensorMetrics.from_dict(m) for n, m in data["tensors"].items()},
            groups=tuple(
                GroupMetrics(**{**g, "tensors": tuple(g["tensors"])}) for g in data["groups"]
            ),
            control=(SourceInfo(**control[0]), SourceInfo(**control[1])) if control else None,
            schema_version=data["schema_version"],
            metrics_version=data["metrics_version"],
            tool_version=data["tool_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> DiffResult:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class DiffOptions:
    cache_dir: Path | None = None  # None disables the metrics cache
    # Reference-scale pair (not a null; see noise.py): paths or CheckpointSources.
    control: tuple[Any, Any] | None = None


# -- hashing and cache --------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


# sha256 of local files already hashed in this process, keyed by (path, size, mtime_ns), so a
# checkpoint used by several pairs (a trajectory, a control pair) is read for hashing once.
_HASH_MEMO: dict[tuple[str, int, int], str] = {}


def local_source(path: str | Path) -> LocalSource:
    """Hash a local file (streaming, memoized per run) and describe it."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"Checkpoint not found: {path}")
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    sha = _HASH_MEMO.get(key)
    if sha is None:
        start = time.perf_counter()
        sha = _HASH_MEMO[key] = sha256_file(path)
        log.debug("sha256 %s = %s (%.2fs)", path, sha[:12], time.perf_counter() - start)
    label = path.parent.name if path.name == "model.safetensors" else path.stem
    info = SourceInfo(label=label, path=str(path), sha256=sha, size_bytes=stat.st_size)
    return LocalSource(path, info)


def _as_source(item: Any) -> CheckpointSource:
    return local_source(item) if isinstance(item, str | Path) else item


def cache_path(cache_dir: Path, sha_a: str, sha_b: str) -> Path:
    return cache_dir / f"{sha_a}_{sha_b}_m{metrics.METRICS_VERSION}.json"


def files_needed(
    a: CheckpointSource, b: CheckpointSource, cache_dir: Path | None
) -> list[CheckpointSource]:
    """Which of the two sources must be fetched to diff them (for download planning)."""
    if a.info.sha256 == b.info.sha256:
        return [a]  # identical: only A's header is read
    if cache_dir is not None and _load_cache(cache_path(cache_dir, a.info.sha256, b.info.sha256)):
        return []
    return [a, b]


def _load_cache(path: Path) -> tuple[dict[str, TensorMeasurement], list[str]] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["cache_format"] != CACHE_FORMAT or data["metrics_version"] != (
            metrics.METRICS_VERSION
        ):
            return None
        measured = {n: TensorMeasurement.from_dict(m) for n, m in data["measurements"].items()}
        return measured, list(data["skipped"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Ignoring unreadable metrics cache %s (%s)", path, exc)
        return None


def _save_cache(
    path: Path,
    sha_a: str,
    sha_b: str,
    measured: Mapping[str, TensorMeasurement],
    skipped: list[str],
) -> None:
    payload = {
        "cache_format": CACHE_FORMAT,
        "metrics_version": metrics.METRICS_VERSION,
        "sha256_a": sha_a,
        "sha256_b": sha_b,
        "skipped": skipped,
        "measurements": {n: m.to_dict() for n, m in measured.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


# -- measurement --------------------------------------------------------------------------
def _measure_pair(
    a: CheckpointSource, b: CheckpointSource, cache_dir: Path | None
) -> tuple[dict[str, TensorMeasurement], list[str]]:
    """Raw measurements for every shared parameter, from the cache when possible.

    Only on a cache miss are the two files fetched (downloaded if needed); both stay pinned
    until every tensor has been measured.
    """
    cpath = cache_path(cache_dir, a.info.sha256, b.info.sha256) if cache_dir else None
    if cpath is not None:
        cached = _load_cache(cpath)
        if cached is not None:
            log.info("Metrics cache hit: %s", cpath.name)
            return cached
        log.info("Metrics cache miss: computing %s vs %s", a.info.label, b.info.label)

    measured: dict[str, TensorMeasurement] = {}
    with (
        a.fetch() as path_a,
        b.fetch() as path_b,
        Checkpoint(path_a) as ca,
        Checkpoint(path_b) as cb,
    ):
        start = time.perf_counter()
        specs_a, specs_b = ca.specs(), cb.specs()
        names = check_compatible(specs_a, specs_b)
        skipped = sorted(set(parameter_specs(specs_a)[1]) | set(parameter_specs(specs_b)[1]))
        for name in names:
            t = time.perf_counter()
            try:
                measured[name] = metrics.measure_tensor(
                    ca.load(name), cb.load(name), specs_a[name].dtype, specs_b[name].dtype
                )
            except ValueError as exc:
                raise CheckpointFormatError(f"Tensor {name!r}: {exc}") from exc
            log.debug("measured %s in %.2fs", name, time.perf_counter() - t)
    log.info("Measured %d tensors in %.1fs", len(measured), time.perf_counter() - start)
    if cpath is not None:
        _save_cache(cpath, a.info.sha256, b.info.sha256, measured, skipped)
    return measured, skipped


def _group(
    items: Mapping[str, Any],
    floor_of: Callable[[str], float],
    control_of: Callable[[int | None, str], float | None],
) -> tuple[GroupMetrics, ...]:
    """Aggregate measurements (or TensorMetrics) by (layer, component) via sums of squares."""
    buckets: dict[tuple[int | None, str], list[str]] = {}
    for name in items:
        key = classify(name)
        buckets.setdefault((key.layer, key.component), []).append(name)

    groups = []
    for (layer, component), names in buckets.items():
        ss_a = sum(items[n].norm_a ** 2 for n in names)
        ss_d = sum(items[n].abs_delta ** 2 for n in names)
        abs_delta, norm_a = math.sqrt(ss_d), math.sqrt(ss_a)
        if norm_a > 0.0:
            rel: float | None = abs_delta / norm_a
            floor = math.sqrt(sum((floor_of(n) * items[n].norm_a) ** 2 for n in names)) / norm_a
        else:
            rel = 0.0 if abs_delta == 0.0 else None
            floor = max(floor_of(n) for n in names)
        control = control_of(layer, component)
        groups.append(
            GroupMetrics(
                layer=layer,
                component=component,
                tensors=tuple(names),
                norm_a=norm_a,
                abs_delta=abs_delta,
                rel_delta=rel,
                floor=floor,
                control=control,
                status=noise.assess(rel, abs_delta, floor, control),
            )
        )
    return tuple(groups)


def _identical_result(a: CheckpointSource, b: SourceInfo, control: Any) -> DiffResult:
    """Byte-identical files: build the grid from A's header only; measure nothing."""
    with a.fetch() as path_a, Checkpoint(path_a) as ca:
        params, skipped = parameter_specs(ca.specs())
    buckets: dict[tuple[int | None, str], list[str]] = {}
    for name in params:
        key = classify(name)
        buckets.setdefault((key.layer, key.component), []).append(name)
    groups = tuple(
        GroupMetrics(
            layer=layer,
            component=component,
            tensors=tuple(names),
            norm_a=None,
            abs_delta=0.0,
            rel_delta=0.0,
            floor=max(noise.dtype_floor(params[n].dtype, params[n].dtype) for n in names),
            control=None,
            status=noise.NO_CHANGE,
        )
        for (layer, component), names in buckets.items()
    )
    return DiffResult(
        a=a.info,
        b=b,
        identical=True,
        n_tensors=len(params),
        skipped=tuple(sorted(skipped)),
        tensors={},
        groups=groups,
        control=control,
    )


def _check_control_matches(
    measured: Mapping[str, TensorMeasurement], control: Mapping[str, TensorMeasurement]
) -> None:
    shape_mismatches = [
        (n, measured[n].shape, control[n].shape)
        for n in measured
        if n in control and measured[n].shape != control[n].shape
    ]
    only_main = [n for n in measured if n not in control]
    only_control = [n for n in control if n not in measured]
    if only_main or only_control or shape_mismatches:
        raise ArchitectureMismatch(
            only_main,
            only_control,
            shape_mismatches,
            header="The control pair does not have the same architecture as the compared pair.",
            labels=("the compared pair", "the control pair"),
        )


# -- public API ---------------------------------------------------------------------------
def diff_checkpoints(a: Any, b: Any, options: DiffOptions | None = None) -> DiffResult:
    """Compare two same-architecture checkpoints. B is measured relative to A.

    ``a``, ``b`` and the control pair are local paths or CheckpointSources.
    """
    options = options or DiffOptions()
    a, b = _as_source(a), _as_source(b)
    control = tuple(_as_source(c) for c in options.control) if options.control else None
    control_info = (control[0].info, control[1].info) if control else None

    if a.info.sha256 == b.info.sha256:
        log.info("Files are byte-identical (sha256 %s): no difference", a.info.sha256[:12])
        return _identical_result(a, b.info, control_info)

    # The compared pair first: measured and cached before any control file is fetched.
    measured, skipped = _measure_pair(a, b, options.cache_dir)

    control_tensors: dict[str, float | None] = {}
    control_groups: dict[tuple[int | None, str], float | None] = {}
    if control and control[0].info.sha256 != control[1].info.sha256:
        c_measured, _ = _measure_pair(control[0], control[1], options.cache_dir)
        _check_control_matches(measured, c_measured)
        control_tensors = {n: m.rel_delta for n, m in c_measured.items()}
        for g in _group(c_measured, lambda _n: 0.0, lambda _l, _c: None):
            control_groups[(g.layer, g.component)] = g.rel_delta
    # A byte-identical control pair gives a zero reference scale: no extra muting.

    assessed = {n: assess_tensor(m, control_tensors.get(n)) for n, m in measured.items()}
    groups = _group(
        assessed,
        lambda n: assessed[n].floor,
        lambda layer, component: control_groups.get((layer, component)),
    )
    return DiffResult(
        a=a.info,
        b=b.info,
        identical=False,
        n_tensors=len(assessed),
        skipped=tuple(skipped),
        tensors=assessed,
        groups=groups,
        control=control_info,
    )

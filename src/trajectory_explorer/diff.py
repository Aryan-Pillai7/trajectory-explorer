"""Diff engine: two checkpoint files -> DiffResult (per tensor + per (layer, component) group).

Flow: stream-hash both files -> byte-identical short-circuit -> raw measurements from the
JSON cache or computed one tensor at a time -> floor/control/labels applied -> groups.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    label: str
    path: str
    sha256: str
    size_bytes: int


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
    control: tuple[Path, Path] | None = None  # reference-scale pair (not a null; see noise.py)


# -- hashing and cache --------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> SourceInfo:
    if not path.is_file():
        raise InputError(f"Checkpoint not found: {path}")
    start = time.perf_counter()
    sha = sha256_file(path)
    log.debug("sha256 %s = %s (%.2fs)", path, sha[:12], time.perf_counter() - start)
    label = path.parent.name if path.name == "model.safetensors" else path.stem
    return SourceInfo(label=label, path=str(path), sha256=sha, size_bytes=path.stat().st_size)


def cache_path(cache_dir: Path, sha_a: str, sha_b: str) -> Path:
    return cache_dir / f"{sha_a}_{sha_b}_m{metrics.METRICS_VERSION}.json"


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
    a: SourceInfo, b: SourceInfo, cache_dir: Path | None
) -> tuple[dict[str, TensorMeasurement], list[str]]:
    """Raw measurements for every shared parameter, from the cache when possible."""
    cpath = cache_path(cache_dir, a.sha256, b.sha256) if cache_dir else None
    if cpath is not None:
        cached = _load_cache(cpath)
        if cached is not None:
            log.info("Metrics cache hit: %s", cpath.name)
            return cached
        log.info("Metrics cache miss: computing %s vs %s", a.label, b.label)

    start = time.perf_counter()
    measured: dict[str, TensorMeasurement] = {}
    with Checkpoint(a.path) as ca, Checkpoint(b.path) as cb:
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
        _save_cache(cpath, a.sha256, b.sha256, measured, skipped)
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


def _identical_result(a: SourceInfo, b: SourceInfo, control: Any) -> DiffResult:
    """Byte-identical files: build the grid from the header only; measure nothing."""
    with Checkpoint(a.path) as ca:
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
        a=a,
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
def diff_checkpoints(
    path_a: str | Path, path_b: str | Path, options: DiffOptions | None = None
) -> DiffResult:
    """Compare two same-architecture safetensors files. B is measured relative to A."""
    options = options or DiffOptions()
    a, b = _source(Path(path_a)), _source(Path(path_b))

    control_info = None
    c_measured: dict[str, TensorMeasurement] = {}
    control_tensors: dict[str, float | None] = {}
    control_groups: dict[tuple[int | None, str], float | None] = {}
    if options.control is not None:
        ca, cb = _source(Path(options.control[0])), _source(Path(options.control[1]))
        control_info = (ca, cb)
        if ca.sha256 != cb.sha256:
            c_measured, _ = _measure_pair(ca, cb, options.cache_dir)
            control_tensors = {n: m.rel_delta for n, m in c_measured.items()}
            for g in _group(c_measured, lambda _n: 0.0, lambda _l, _c: None):
                control_groups[(g.layer, g.component)] = g.rel_delta
        # A byte-identical control pair gives a zero reference scale: no extra muting.

    if a.sha256 == b.sha256:
        log.info("Files are byte-identical (sha256 %s): no difference", a.sha256[:12])
        return _identical_result(a, b, control_info)

    measured, skipped = _measure_pair(a, b, options.cache_dir)
    if c_measured:
        _check_control_matches(measured, c_measured)

    assessed = {n: assess_tensor(m, control_tensors.get(n)) for n, m in measured.items()}
    groups = _group(
        assessed,
        lambda n: assessed[n].floor,
        lambda layer, component: control_groups.get((layer, component)),
    )
    return DiffResult(
        a=a,
        b=b,
        identical=False,
        n_tensors=len(assessed),
        skipped=tuple(skipped),
        tensors=assessed,
        groups=groups,
        control=control_info,
    )

"""Trajectory: an ordered list of checkpoints -> one DiffResult per adjacent interval.

Every interval goes through the normal diff engine and its pair cache. Checkpoints are
fetched in order, so with a two-file rolling store each one is downloaded exactly once:
interval i needs checkpoints i and i+1, and checkpoint i is still on disk from interval i-1.
Adjacent-step diffs are a reference scale, not a null (decision D12).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_explorer import metrics
from trajectory_explorer._version import __version__
from trajectory_explorer.arch import COMPONENTS
from trajectory_explorer.diff import (
    CheckpointSource,
    DiffOptions,
    DiffResult,
    SourceInfo,
    diff_checkpoints,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # v2: precision-rule fields (D40)
DEFAULT_POINTS = 25
_STEP_BRANCH = re.compile(r"^step(\d+)$")
_STEP_LABEL = re.compile(r"(?:^|@)step(\d+)$")


# -- step selection -----------------------------------------------------------------------
def step_branches(names: Sequence[str]) -> list[int]:
    """Step numbers of branches named stepN, sorted and unique."""
    return sorted({int(m.group(1)) for n in names if (m := _STEP_BRANCH.match(n))})


def select_steps(available: Sequence[int], target: int = DEFAULT_POINTS) -> list[int]:
    """About ``target`` log-spaced steps (log of step+1), snapped to available ones.

    Always includes the first and the last step. Snapping can map two targets to one step, so
    the number of targets grows until ``target`` distinct steps are chosen.
    """
    steps = sorted(set(available))
    if len(steps) <= target:
        return steps
    logs = np.log1p(np.asarray(steps, dtype=np.float64))
    chosen: set[int] = set()
    for n in range(target, len(steps) + 1):
        targets = np.linspace(logs[0], logs[-1], n)
        chosen = {int(np.argmin(np.abs(logs - t))) for t in targets}
        if len(chosen) >= target:
            break
    return [steps[i] for i in sorted(chosen)]


def step_of(info: SourceInfo) -> int | None:
    """The training step of a checkpoint whose label ends in stepN (Hub or folder name)."""
    match = _STEP_LABEL.search(info.label)
    return int(match.group(1)) if match else None


# -- result -------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryResult:
    points: tuple[SourceInfo, ...]
    intervals: tuple[DiffResult, ...]  # intervals[i] compares points[i] -> points[i + 1]
    schema_version: int = SCHEMA_VERSION
    metrics_version: int = field(default_factory=lambda: metrics.METRICS_VERSION)
    tool_version: str = __version__

    @property
    def steps(self) -> list[int | None]:
        return [step_of(p) for p in self.points]

    @property
    def has_steps(self) -> bool:
        steps = self.steps
        return all(s is not None for s in steps) and steps == sorted(steps)

    def interval_labels(self) -> list[str]:
        if self.has_steps:
            s = self.steps
            return [f"{s[i]} to {s[i + 1]}" for i in range(len(self.intervals))]
        p = self.points
        return [f"{p[i].label} to {p[i + 1].label}" for i in range(len(self.intervals))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics_version": self.metrics_version,
            "tool_version": self.tool_version,
            "points": [asdict(p) for p in self.points],
            "intervals": [r.to_dict() for r in self.intervals],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> TrajectoryResult:
        data = json.loads(text)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema_version {data.get('schema_version')}")
        return cls(
            points=tuple(SourceInfo(**p) for p in data["points"]),
            intervals=tuple(DiffResult.from_dict(r) for r in data["intervals"]),
            schema_version=data["schema_version"],
            metrics_version=data["metrics_version"],
            tool_version=data["tool_version"],
        )


def run_trajectory(
    sources: Sequence[CheckpointSource], cache_dir: Path | None = None
) -> TrajectoryResult:
    """Diff each adjacent pair, in order, through the normal engine and pair cache."""
    if len(sources) < 2:
        raise ValueError("a trajectory needs at least two checkpoints")
    intervals = []
    for i, (a, b) in enumerate(itertools.pairwise(sources)):
        log.info("Interval %d/%d: %s to %s", i + 1, len(sources) - 1, a.info.label, b.info.label)
        intervals.append(diff_checkpoints(a, b, DiffOptions(cache_dir=cache_dir)))
    return TrajectoryResult(points=tuple(s.info for s in sources), intervals=tuple(intervals))


# -- aggregation for the report -----------------------------------------------------------
VECTORS = "vectors"  # the line chart's single series for all 1-D tensors (D41)
_KIND_ORDER = {"matrix": 0, "vector": 1}


def row_keys(result: TrajectoryResult) -> list[tuple[int | None, str, str]]:
    """(layer, component, kind) rows: embedding first, then each layer (its matrices, then its
    vectors), then the model head."""
    keys = {(g.layer, g.component, g.kind) for r in result.intervals for g in r.groups}
    order = {c: i for i, c in enumerate(COMPONENTS)}

    def sort_key(k: tuple[int | None, str, str]) -> tuple[int, int, int]:
        return (-1 if k[0] is None else k[0], _KIND_ORDER[k[2]], order[k[1]])

    embed = sorted((k for k in keys if k[0] is None and k[1] == "embed"), key=sort_key)
    head = sorted((k for k in keys if k[0] is None and k[1] != "embed"), key=sort_key)
    layers = sorted((k for k in keys if k[0] is not None), key=sort_key)
    return embed + layers + head


def component_series(result: TrajectoryResult) -> dict[str, list[float | None]]:
    """Per matrix component, plus one series for all vectors, the relative change per interval.

    sqrt(sum ||dW||^2) / sqrt(sum ||W||^2) over every layer's matrices of that component, and
    over every vector (bias, norm scale) for the VECTORS series, so a tiny bias never drives
    a matrix line (D41). Byte-identical intervals count as 0.0; a series starting at exactly
    zero gives None.
    """
    groups_all = [g for r in result.intervals for g in r.groups]
    present = [
        c for c in COMPONENTS if any(g.component == c and g.kind == "matrix" for g in groups_all)
    ]
    if any(g.kind == "vector" for g in groups_all):
        present.append(VECTORS)
    series: dict[str, list[float | None]] = {c: [] for c in present}
    for interval in result.intervals:
        for comp in present:
            if comp == VECTORS:
                groups = [g for g in interval.groups if g.kind == "vector"]
            else:
                groups = [g for g in interval.groups if g.component == comp and g.kind == "matrix"]
            if interval.identical or not groups:
                series[comp].append(0.0 if groups else None)
                continue
            ss_a = sum((g.norm_a or 0.0) ** 2 for g in groups)
            ss_d = sum(g.abs_delta**2 for g in groups)
            if ss_a > 0:
                series[comp].append(math.sqrt(ss_d) / math.sqrt(ss_a))
            else:
                series[comp].append(0.0 if ss_d == 0 else None)
    return series

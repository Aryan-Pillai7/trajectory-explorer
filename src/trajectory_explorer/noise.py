"""Noise floor, significance status and labels for one tensor delta.

The true null here is floating-point rounding: two stored copies of the same weights can differ
by up to the unit roundoff of their dtypes. The floor is ``k * sqrt(uA^2 + uB^2) / sqrt(3)``,
where ``u / sqrt(3)`` is a conservative RMS relative rounding error per element (the exact
value for round-to-nearest is about 0.42 u) and ``k = 3`` is a safety factor.

An optional *control* value (e.g. the relative delta between two adjacent training
checkpoints) is a reference scale, not a null: real learning happens between adjacent
checkpoints. Deltas at or below it are reported as "below_control", separately from
"below_floor".
"""

from __future__ import annotations

import math

UNIT_ROUNDOFF: dict[str, float] = {
    "F64": 2.0**-53,
    "F32": 2.0**-24,
    "F16": 2.0**-11,
    "BF16": 2.0**-8,
}
FLOOR_K = 3.0

# Heuristic label thresholds (documented in docs/metrics.md, chunk 11).
LOW_RANK_RATIO = 0.25  # erank below 25% of a same-shape Gaussian matrix's erank => low-rank
TOP_ROW_FRACTION = 0.05  # concentration looks at the top 5% of rows by delta energy...
CONCENTRATED_SHARE = 0.5  # ...and calls the delta concentrated if they hold >= 50% of it
MIN_ROWS_FOR_CONCENTRATION = 20  # below this, "top 5% of rows" is less than one row

NO_CHANGE = "no_change"
FROM_ZERO = "from_zero"
BELOW_FLOOR = "below_floor"
BELOW_CONTROL = "below_control"
SIGNIFICANT = "significant"
SIGNIFICANT_STATUSES = frozenset({SIGNIFICANT, FROM_ZERO})


def dtype_floor(dtype_a: str, dtype_b: str, k: float = FLOOR_K) -> float:
    """Relative-delta floor explained by rounding the two tensors to their stored dtypes."""
    try:
        ua, ub = UNIT_ROUNDOFF[dtype_a], UNIT_ROUNDOFF[dtype_b]
    except KeyError as exc:
        raise ValueError(f"no rounding model for dtype {exc.args[0]!r}") from None
    return k * math.sqrt(ua * ua + ub * ub) / math.sqrt(3.0)


def assess(
    rel_delta: float | None, abs_delta: float, floor: float, control: float | None = None
) -> str:
    """Classify a delta. ``rel_delta`` is None when the reference tensor is exactly zero."""
    if abs_delta == 0.0:
        return NO_CHANGE
    if rel_delta is None:
        # Moving away from an exact zero (e.g. a bias at step 0) is never rounding noise.
        return FROM_ZERO
    if rel_delta <= floor:
        return BELOW_FLOOR
    if control is not None and rel_delta <= control:
        return BELOW_CONTROL
    return SIGNIFICANT


def rank_label(erank: float | None, erank_random: float | None) -> str:
    if erank is None or not erank_random:
        return "n/a"
    return "low-rank" if erank / erank_random < LOW_RANK_RATIO else "dense"


def concentration_label(top_row_share: float | None, n_rows: int) -> str:
    if top_row_share is None or n_rows < MIN_ROWS_FOR_CONCENTRATION:
        return "n/a"
    return "concentrated" if top_row_share >= CONCENTRATED_SHARE else "spread"

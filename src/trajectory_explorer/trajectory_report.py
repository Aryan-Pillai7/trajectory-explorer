"""Render a TrajectoryResult as one self-contained, offline HTML page with inline SVG.

Same rules as the pair report (report.py): no JavaScript, no external requests, escaped text,
light/dark/print styles. The heatmap has two panels (weight matrices, vectors), each with its
own log-spaced colour scale fixed across every interval (the same validated blue ramp), so
columns within a panel are comparable; the panels are not comparable with each other (D46).
The line chart uses the validated categorical palette in fixed slot order (dataviz validator:
all adjacent checks pass in both modes; three light slots are under 3:1 contrast, so direct
labels and the interval table are the required relief), with a different marker and dash per
series so identity never rests on colour alone.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from html import escape
from string import Template

from trajectory_explorer import noise
from trajectory_explorer._version import __version__
from trajectory_explorer.diff import GroupMetrics
from trajectory_explorer.report import (
    BASE_CSS,
    COMPONENT_LABELS,
    FROM_ZERO_MARK,
    N_BINS,
    PANEL_NOTE,
    PANEL_TITLES,
    STATUS_TEXT,
    _bin,
    pct,
    precision_rule_note,
    sci,
    vector_share_lines,
)
from trajectory_explorer.report import _legend as bin_legend
from trajectory_explorer.trajectory import (
    VECTORS,
    TrajectoryResult,
    component_series,
    row_keys,
)

SERIES_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SERIES_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
MARKERS = ["circle", "square", "triangle", "diamond", "triangle_down", "plus", "cross", "circle"]
DASHES = ["", "7 4", "2 3"]


def _vars(colors: list[str]) -> str:
    return " ".join(f"--c{i}: {c};" for i, c in enumerate(colors))


EXTRA_CSS = (
    f"""
:root {{ {_vars(SERIES_LIGHT)} }}
@media (prefers-color-scheme: dark) {{ :root {{ {_vars(SERIES_DARK)} }} }}
@media print {{ :root {{ {_vars(SERIES_LIGHT)} }} }}
"""
    + "".join(
        f".s{i} {{ stroke: var(--c{i}); }} .m{i} {{ fill: var(--c{i}); }}\n" for i in range(8)
    )
    + """
svg.lines { max-width: 100%; height: auto; font-size: 12px; }
.axis { stroke: var(--hair); stroke-width: 1; }
.grid { stroke: var(--hair); stroke-width: 1; }
.tick { fill: var(--muted); font-variant-numeric: tabular-nums; }
.axis-title { fill: var(--ink-2); }
.series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.marker { stroke: var(--surface); stroke-width: 2; }
.leader { stroke: var(--muted); stroke-width: 1; }
.end-label { fill: var(--ink); }
.floor-line { stroke: var(--muted); stroke-width: 1; }
.col-head { fill: var(--ink-2); font-size: 11px; }
svg.traj-heatmap { font-size: 11px; height: auto; display: block; }
svg.traj-heatmap .cell-text.fz { font-size: 10px; }
.lead-note { margin: 10px 2px 0; color: var(--ink-2); font-size: 14px; }
.lead-note a { color: inherit; }
"""
)


SERIES_LABELS = {**COMPONENT_LABELS, VECTORS: "all vectors"}
MAX_SERIES = 7  # validated palette slots (dataviz validator, light and dark)


def _where(layer: int | None, component: str, kind: str | None = None) -> str:
    label = COMPONENT_LABELS.get(component, component)
    where = f"layer {layer} {label}" if layer is not None else label
    return f"{where} ({kind})" if kind else where


def _row_label(layer: int | None, component: str, kind: str) -> str:
    label = COMPONENT_LABELS.get(component, component)
    return (f"L{layer} {label}" if layer is not None else label) + f" ({kind})"


# -- banner -------------------------------------------------------------------------------
def summary_sentence(result: TrajectoryResult) -> tuple[bool, str]:
    labels = result.interval_labels()
    unit = "steps" if result.has_steps else "interval"
    cells = [(i, g) for i, r in enumerate(result.intervals) for g in r.groups]
    sig = [(i, g) for i, g in cells if g.significant]
    if not sig:
        return False, (
            f"No significant change across the sampled steps: all {len(cells)} cells over "
            f"{len(result.intervals)} intervals are at or below the noise floor."
        )
    best: dict[tuple[str, str], tuple[float, int, GroupMetrics]] = {}
    for i, g in sig:
        score = g.rel_delta if g.rel_delta is not None else -1.0  # from-zero ranks last
        key = (g.component, g.kind)
        if key not in best or score > best[key][0]:
            best[key] = (score, i, g)
    # Matrices first: a tiny vector with a huge relative change must not headline (D41).
    ranked = sorted(best.items(), key=lambda kv: (kv[0][1] == "matrix", kv[1][0]), reverse=True)
    ((c1, k1), (_, i1, g1)), rest = ranked[0], ranked[1:]
    change = "moved away from zero" if g1.rel_delta is None else f"{pct(g1.rel_delta)}"
    text = (
        f"{COMPONENT_LABELS.get(c1, c1)} ({k1}) moved most (largest: "
        f"{_where(g1.layer, g1.component, g1.kind)} in {unit} {labels[i1]}, {change})"
    )
    if rest:
        (c2, k2), (_, i2, _g2) = rest[0]
        text += f", followed by {COMPONENT_LABELS.get(c2, c2)} ({k2}) (peak in {unit} {labels[i2]})"
    return True, text + f". {len(sig)} of {len(cells)} cells are above the noise floor."


# -- heatmap ------------------------------------------------------------------------------
_LABEL_W, _HEAD_H, _CELL_H, _GAP, _LAYER_GAP, _RIGHT_PAD = 170, 92, 16, 2, 6, 60
# Width budget: the usable width inside a report section at a 1280 px viewport (main is at
# most 1100 px, minus page and section padding). Cells shrink to fit it; below MIN_CELL_W a
# cell is too small to read or hover reliably, so the panel keeps MIN_CELL_W and scrolls.
# With these numbers everything fits up to 42 intervals; from 43 on the panel scrolls (D49).
_TARGET_W, _MAX_CELL_W, _MIN_CELL_W = 1000, 40.0, 16.0


def heatmap_geometry(n_intervals: int) -> tuple[float, float, bool]:
    """(cell width, SVG width, scrolls?) for a panel with this many columns."""
    fit = (_TARGET_W - _LABEL_W - _RIGHT_PAD) / max(n_intervals, 1) - _GAP
    cell = min(_MAX_CELL_W, fit)
    scrolls = cell < _MIN_CELL_W
    if scrolls:
        cell = _MIN_CELL_W
    return cell, _LABEL_W + n_intervals * (cell + _GAP) + _RIGHT_PAD, scrolls


def _global_edges(result: TrajectoryResult, kind: str) -> list[float] | None:
    """One scale per kind, fixed across ALL intervals so columns stay comparable (D37, D46)."""
    values = [
        g.rel_delta
        for r in result.intervals
        for g in r.groups
        if g.kind == kind and g.status == noise.SIGNIFICANT and g.rel_delta
    ]
    if not values:
        return None
    lo, hi = math.log10(min(values)), math.log10(max(values))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    return [10 ** (lo + (hi - lo) * k / N_BINS) for k in range(N_BINS + 1)]


def render_heatmap(result: TrajectoryResult) -> str:
    """Two panels, matrices then vectors, each with its own fixed scale (D46)."""
    parts = [f'<p class="meta">{PANEL_NOTE}</p>']
    for kind in ("matrix", "vector"):
        rows = [k for k in row_keys(result) if k[2] == kind]
        if rows:
            parts.append(
                f'<h3 class="panel">{PANEL_TITLES[kind]}</h3>' + _heatmap_panel(result, rows, kind)
            )
    return "".join(parts)


def _heatmap_panel(
    result: TrajectoryResult, rows: list[tuple[int | None, str, str]], kind: str
) -> str:
    labels = result.interval_labels()
    prefix = "steps " if result.has_steps else ""
    edges = _global_edges(result, kind)
    hatch = f"t-hatch-{kind}"
    lookup = {
        (i, g.layer, g.component, g.kind): g
        for i, r in enumerate(result.intervals)
        for g in r.groups
    }
    ys, y, previous_layer = [], _HEAD_H, "start"
    for layer, _comp, _kind in rows:
        if previous_layer != "start" and layer != previous_layer:
            y += _LAYER_GAP
        ys.append(y)
        y += _CELL_H + _GAP
        previous_layer = layer
    cell_w, width, scrolls = heatmap_geometry(len(labels))
    height = y + 4

    parts = []
    for j, label in enumerate(labels):
        x = _LABEL_W + j * (cell_w + _GAP) + cell_w / 2
        parts.append(
            f'<text class="col-head" x="{x:.1f}" y="{_HEAD_H - 6}" '
            f'transform="rotate(-50 {x:.1f} {_HEAD_H - 6})">{escape(label)}</text>'
        )
    for (layer, comp, kind), y in zip(rows, ys, strict=True):
        parts.append(
            f'<text class="row-label" x="{_LABEL_W - 6}" y="{y + _CELL_H - 4}">'
            f"{escape(_row_label(layer, comp, kind))}</text>"
        )
        for j, label in enumerate(labels):
            g = lookup.get((j, layer, comp, kind))
            if g is None:
                continue
            x = _LABEL_W + j * (cell_w + _GAP)
            if g.status == noise.SIGNIFICANT and edges and g.rel_delta:
                cls, fill = f"sig bin{_bin(g.rel_delta, edges)}", ""
            elif g.status == noise.FROM_ZERO:  # own style, never the top bin (D50)
                cls, fill = "fromzero", ""
            else:
                cls, fill = f"muted {g.status}", f' fill="url(#{hatch})"'
            reason = STATUS_TEXT.get(g.status, g.status)
            if result.intervals[j].identical:
                reason = "no change: the two files are byte-identical"
            tip = (
                f"{_where(layer, comp, kind)}, {prefix}{label}\n"
                f"relative change {pct(g.rel_delta)} (||dW||/||W|| = {sci(g.rel_delta)})\n"
                f"noise floor {sci(g.floor)}\nstatus: {reason}"
            )
            shares = vector_share_lines(g, result.intervals[j].tensors)
            if shares:
                tip += "\n" + "\n".join(shares)
            parts.append(
                f'<g class="cell {cls}"><title>{escape(tip)}</title><rect x="{x:.1f}" y="{y}" '
                f'width="{cell_w:.1f}" height="{_CELL_H}" rx="2"{fill}/>'
                + (
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + _CELL_H - 4}" '
                    f'class="cell-text fz">{FROM_ZERO_MARK}</text>'
                    if cls == "fromzero"
                    else ""
                )
                + "</g>"
            )
    n_sig = sum(1 for r in result.intervals for g in r.groups if g.significant and g.kind == kind)
    n_all = sum(1 for r in result.intervals for g in r.groups if g.kind == kind)
    aria = (
        "Heatmap of relative change per layer, component and interval: "
        f"{n_sig} of {n_all} cells above the noise floor."
    )
    return (
        f'<svg class="traj-heatmap" id="t-heatmap-{kind}" viewBox="0 0 {width:.0f} {height}" '
        # Fits: scale to the container (never wider than drawn). Too many columns: keep a
        # readable minimum width and let the surrounding .scroll box scroll sideways.
        + (
            f'width="{width:.0f}" style="min-width: {width:.0f}px" '
            if scrolls
            else f'width="100%" style="max-width: {width:.0f}px" '
        )
        + f'role="img" aria-label="{escape(PANEL_TITLES[kind] + ": " + aria)}">'
        f'<defs><pattern id="{hatch}" width="6" height="6" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" '
        'class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern>'
        "</defs>"
        + "".join(parts)
        + "</svg>"
        + bin_legend(edges, f"t-hatch-key-{kind}", PANEL_TITLES[kind])
    )


# -- line chart ---------------------------------------------------------------------------
_W, _H, _ML, _MR, _MT, _MB = 860, 380, 70, 170, 16, 52


def _marker(shape: str, x: float, y: float, cls: str, r: float = 4.5) -> str:
    if shape == "circle":
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if shape == "square":
        return (
            f'<rect class="{cls}" x="{x - r:.1f}" y="{y - r:.1f}" '
            f'width="{2 * r}" height="{2 * r}"/>'
        )
    if shape in ("triangle", "triangle_down"):
        s = 1 if shape == "triangle" else -1
        tip, base = y - s * r * 1.2, y + s * r * 0.8
        pts = f"{x:.1f},{tip:.1f} {x - r * 1.1:.1f},{base:.1f} {x + r * 1.1:.1f},{base:.1f}"
        return f'<polygon class="{cls}" points="{pts}"/>'
    if shape == "diamond":
        top, bottom, left, right = y - r * 1.3, y + r * 1.3, x - r * 1.1, x + r * 1.1
        pts = f"{x:.1f},{top:.1f} {right:.1f},{y:.1f} {x:.1f},{bottom:.1f} {left:.1f},{y:.1f}"
        return f'<polygon class="{cls}" points="{pts}"/>'
    w = r * 0.45  # plus / cross drawn as a thick polygon so it can be filled like the others
    arm = [
        (-r, -w),
        (-w, -w),
        (-w, -r),
        (w, -r),
        (w, -w),
        (r, -w),
        (r, w),
        (w, w),
        (w, r),
        (-w, r),
        (-w, w),
        (-r, w),
    ]
    if shape == "cross":
        c = math.cos(math.pi / 4)
        arm = [((px - py) * c, (px + py) * c) for px, py in arm]
    pts = " ".join(f"{x + px:.1f},{y + py:.1f}" for px, py in arm)
    return f'<polygon class="{cls}" points="{pts}"/>'


def _tick_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def render_lines(result: TrajectoryResult) -> tuple[str, str]:
    """(svg, legend html). x: interval end step on a log axis (or interval number)."""
    series = component_series(result)
    omitted = list(series)[MAX_SERIES:]  # e.g. "other" matrices of an unknown architecture
    series = dict(list(series.items())[:MAX_SERIES])
    labels = result.interval_labels()
    prefix = "steps " if result.has_steps else ""
    n = len(result.intervals)
    if result.has_steps:
        ends = [s for s in result.steps[1:] if s is not None]
        xs = [math.log10(max(e, 1)) for e in ends]
        x_title = "end step of the interval (log scale)"
    else:
        xs = [float(i + 1) for i in range(n)]
        x_title = "interval number"
    x_lo, x_hi = min(xs), max(xs)
    if x_hi - x_lo < 1e-9:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5

    floor_ref = max((g.floor for r in result.intervals for g in r.groups), default=0.0)
    values = [v for vs in series.values() for v in vs if v]
    if not values:
        return "<p>No interval has a measurable change to plot.</p>", ""
    y_lo = math.floor(math.log10(min(values)))
    # Show the floor line only if it is near the data; otherwise say where it is instead of
    # stretching the axis over empty decades.
    floor_shown = bool(floor_ref) and floor_ref >= 10.0 ** (y_lo - 2)
    if floor_shown:
        y_lo = min(y_lo, math.floor(math.log10(floor_ref)))
    y_hi = math.ceil(math.log10(max(values)))
    if y_hi <= y_lo:
        y_hi = y_lo + 1

    def px(xv: float) -> float:
        return _ML + (xv - x_lo) / (x_hi - x_lo) * (_W - _ML - _MR)

    def py(v: float) -> float:
        return _MT + (y_hi - math.log10(v)) / (y_hi - y_lo) * (_H - _MT - _MB)

    parts = []
    step = 1 if y_hi - y_lo <= 8 else 2
    for k in range(y_lo, y_hi + 1, step):
        yy = py(10.0**k)
        parts.append(f'<line class="grid" x1="{_ML}" x2="{_W - _MR}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{_ML - 6}" y="{yy + 4:.1f}" text-anchor="end">'
            f"{pct(10.0**k)}</text>"
        )
    if result.has_steps:
        for k in range(math.ceil(x_lo), math.floor(x_hi) + 1):
            xx = px(float(k))
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{_H - _MB + 16}" text-anchor="middle">'
                f"{_tick_label(10.0**k)}</text>"
            )
    else:
        for i, xv in enumerate(xs):
            parts.append(
                f'<text class="tick" x="{px(xv):.1f}" y="{_H - _MB + 16}" '
                f'text-anchor="middle">{i + 1}</text>'
            )
    parts.append(f'<line class="axis" x1="{_ML}" x2="{_W - _MR}" y1="{_H - _MB}" y2="{_H - _MB}"/>')
    parts.append(
        f'<text class="axis-title" x="{(_ML + _W - _MR) / 2}" y="{_H - 8}" '
        f'text-anchor="middle">{x_title}</text>'
    )
    parts.append(
        f'<text class="axis-title" transform="rotate(-90)" x="{-(_MT + _H - _MB) / 2}" y="16" '
        'text-anchor="middle">relative change per interval (log scale)</text>'
    )
    if floor_shown:
        fy = py(floor_ref)
        parts.append(
            f'<line class="floor-line" x1="{_ML}" x2="{_W - _MR}" y1="{fy:.1f}" y2="{fy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{_ML + 4}" y="{fy - 4:.1f}">'
            f"rounding floor {sci(floor_ref)}</text>"
        )

    ends = []
    legend = []
    for s, (comp, vs) in enumerate(series.items()):
        name = SERIES_LABELS.get(comp, comp)
        dash = DASHES[s % len(DASHES)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = [(px(xs[i]), py(v), i, v) for i, v in enumerate(vs) if v]
        runs, run, prev = [], [], None
        for p in points:
            if prev is not None and p[2] != prev + 1:
                runs.append(run)
                run = []
            run.append(p)
            prev = p[2]
        runs.append(run)
        for r in runs:
            if len(r) > 1:
                coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in r)
                parts.append(f'<polyline class="series s{s}" points="{coords}"{dash_attr}/>')
        for x, y, i, v in points:
            tip = escape(f"{name}, {prefix}{labels[i]}: {pct(v)} relative change")
            parts.append(f"<g><title>{tip}</title>{_marker(MARKERS[s], x, y, f'marker m{s}')}</g>")
        if points:
            ends.append({"x": points[-1][0], "y": points[-1][1], "s": s, "name": name})
        key = (
            f'<svg width="34" height="14" aria-hidden="true"><line class="series s{s}" x1="2" '
            f'x2="32" y1="7" y2="7"{dash_attr}/>{_marker(MARKERS[s], 17, 7, f"marker m{s}", 4)}'
            "</svg>"
        )
        legend.append(f'<span class="key">{key}{escape(name)}</span>')

    # Direct labels at the line ends: spread at least 15 px apart, joined by leader lines.
    ends.sort(key=lambda e: e["y"])
    label_ys: list[float] = []
    for e in ends:
        label_ys.append(max(e["y"], label_ys[-1] + 15) if label_ys else e["y"])
    overflow = (label_ys[-1] - (_H - _MB)) if label_ys else 0.0
    if overflow > 0:
        label_ys = [max(_MT, ly - overflow) for ly in label_ys]
    label_x = _W - _MR + 22
    for e, ly in zip(ends, label_ys, strict=True):
        parts.append(
            f'<line class="leader" x1="{e["x"] + 6:.1f}" y1="{e["y"]:.1f}" '
            f'x2="{label_x - 8}" y2="{ly:.1f}"/>'
        )
        parts.append(_marker(MARKERS[e["s"]], label_x, ly, f"marker m{e['s']}", 4))
        parts.append(
            f'<text class="end-label" x="{label_x + 9}" y="{ly + 4:.1f}">{escape(e["name"])}</text>'
        )

    aria = f"Line chart of relative change per interval for {len(series)} components."
    svg = (
        f'<svg class="lines" viewBox="0 0 {_W} {_H}" width="{_W}" role="img" '
        f'aria-label="{escape(aria)}">' + "".join(parts) + "</svg>"
    )
    note = (
        ""
        if floor_shown or not floor_ref
        else f'<p class="meta">The rounding floor ({sci(floor_ref)}) lies far below every '
        "plotted value, so it is not drawn.</p>"
    )
    if omitted:
        names = ", ".join(SERIES_LABELS.get(c, c) for c in omitted)
        note += f'<p class="meta">Not charted (only {MAX_SERIES} series fit): {escape(names)}.</p>'
    return svg, f'<p class="legend">{"".join(legend)}</p>{note}'


# -- tables and text sections ---------------------------------------------------------------
def render_interval_table(result: TrajectoryResult) -> str:
    rows = []
    for label, r in zip(result.interval_labels(), result.intervals, strict=True):
        if r.identical:
            mover, count = "byte-identical files", f"0 of {r.n_tensors}"
        else:
            sig = [g for g in r.groups if g.significant]
            count = f"{len(r.significant_tensors)} of {r.n_tensors}"
            if sig:
                top = max(
                    sig,
                    key=lambda g: (g.kind == "matrix", g.rel_delta is not None, g.rel_delta or 0),
                )
                mover = f"{_where(top.layer, top.component, top.kind)} ({pct(top.rel_delta)})"
            else:
                mover = "no significant change"
        rows.append(
            f'<tr><td class="nowrap">{escape(label)}</td><td class="num">{count}</td>'
            f"<td>{escape(mover)}</td></tr>"
        )
    return (
        '<table><thead><tr><th>interval</th><th class="num">tensors above floor</th>'
        f"<th>largest change</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_noise_floor(result: TrajectoryResult) -> str:
    floors = Counter(
        (m.dtype_a, m.dtype_b, m.floor) for r in result.intervals for m in r.tensors.values()
    )
    n_rule = sum(m.f16_rule for r in result.intervals for m in r.tensors.values())
    statuses = Counter(g.status for r in result.intervals for g in r.groups)
    n_cells = sum(statuses.values())
    items = (
        "".join(
            f"<li>{escape(a)} &rarr; {escape(b)}: floor {sci(f)} ({pct(f)}), "
            f"{n} tensor comparisons</li>"
            for (a, b, f), n in sorted(floors.items())
        )
        or "<li>No tensors were measured (all intervals byte-identical).</li>"
    )
    above = statuses[noise.SIGNIFICANT] + statuses[noise.FROM_ZERO]
    return (
        "<p>Each column compares two <strong>adjacent</strong> checkpoints. The noise floor is "
        "the smallest relative change that storage rounding alone can produce "
        "(3 &times; &radic;(u<sub>A</sub>&sup2; + u<sub>B</sub>&sup2;) / &radic;3 for the two "
        "formats, where a float32 tensor holding only exact float16 values counts as float16); "
        "cells at or below it are hatched grey.</p>"
        f"<ul>{items}{precision_rule_note(n_rule)}</ul>"
        f"<p>{above} of {n_cells} cells are above the floor, "
        f"{statuses[noise.BELOW_FLOOR]} at or below it, {statuses[noise.NO_CHANGE]} unchanged.</p>"
        "<p><strong>Adjacent-step diffs are a reference scale, not a null.</strong> The model "
        "really learns between neighbouring checkpoints, so a large adjacent change is real "
        "learning and a small one is simply a small step. The only nulls here are rounding "
        "(the floor above) and byte-identical files.</p>"
    )


def render_interval_lengths(result: TrajectoryResult) -> str:
    text = (
        "<p>Intervals have different lengths, so every cell and point is the <em>total</em> "
        "change over its interval, not a change per step. A per-step normalisation "
        "(<code>--normalize</code>) is postponed.</p>"
    )
    if result.has_steps:
        steps = [s for s in result.steps if s is not None]
        lengths = [b - a for a, b in itertools.pairwise(steps)]
        text += (
            f"<p>Here the shortest interval spans {min(lengths):,} step(s) and the longest "
            f"{max(lengths):,} steps (in Pythia's schedule the early checkpoints are 1, 2, 4... "
            "steps apart and the later ones 1000 steps apart).</p>"
        )
    return text


# -- page ---------------------------------------------------------------------------------
_PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="trajectory-explorer $version">
<title>$title</title>
<style>
$css</style>
</head>
<body>
<main>
<h1>$title</h1>
<p class="meta">$n_points checkpoints, $n_intervals adjacent intervals: $first &rarr; $last</p>
<div id="summary" class="banner $banner_cls" role="status">$banner</div>
<p id="reference-note" class="lead-note"><strong>Adjacent-step diffs are a reference scale, not
a null.</strong> Each column compares two neighbouring checkpoints, and the model really learns
between them, so a coloured cell shows how much changed in that interval, not that the change is
unusual. Hatched cells are at or below the rounding floor; columns span different numbers of
steps. More in <a href="#noise-floor">Noise floor</a> and
<a href="#interval-lengths">Interval lengths</a>.</p>

<section id="heatmap">
<h2>Where and when the model changed: (layer, component) &times; interval</h2>
<p class="meta">Relative change ||&Delta;W|| / ||W|| per cell. Each panel's colour scale is fixed
across all intervals, so columns within a panel are comparable.</p>
<div class="scroll">$heatmap</div>
</section>

<section id="lines">
<h2>Relative change per interval, by component</h2>
<p class="meta">Each line aggregates one component's weight matrices over all layers (sums of
squares); "all vectors" is every bias and norm scale together, kept apart so a tiny bias never
drives a matrix line. Both axes are logarithmic.</p>
<div class="scroll">$lines</div>
$line_legend
</section>

<section id="intervals">
<h2>Intervals</h2>
<div class="scroll">$intervals</div>
</section>

<section id="noise-floor">
<h2>Noise floor</h2>
$noise
</section>

<section id="interval-lengths">
<h2>Interval lengths differ</h2>
$lengths
</section>

<footer>Generated by trajectory-explorer $version (metrics v$metrics_version, trajectory schema
v$schema_version). Weight-only comparison: no inputs were run through any model.</footer>
</main>
</body>
</html>
""")


def render_trajectory_html(result: TrajectoryResult) -> str:
    any_sig, banner = summary_sentence(result)
    lines, line_legend = render_lines(result)
    first, last = result.points[0], result.points[-1]
    return _PAGE.substitute(
        version=escape(__version__),
        title=escape(f"Trajectory: {first.label} to {last.label}"),
        n_points=len(result.points),
        n_intervals=len(result.intervals),
        first=escape(first.path),
        last=escape(last.path),
        banner_cls="some" if any_sig else "none",
        banner=escape(banner),
        heatmap=render_heatmap(result),
        lines=lines,
        line_legend=line_legend,
        intervals=render_interval_table(result),
        noise=render_noise_floor(result),
        lengths=render_interval_lengths(result),
        css=BASE_CSS + EXTRA_CSS,
        metrics_version=result.metrics_version,
        schema_version=result.schema_version,
    )

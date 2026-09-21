"""Render a DiffResult as one self-contained, offline HTML page with inline SVG.

No JavaScript, no external requests. Tooltips are SVG <title> elements. Every piece of text
that comes from a checkpoint (tensor names, paths) is HTML-escaped.

Colour: one blue hue, five log-spaced bins (light mode: step 250 -> 650, more change = darker;
dark mode: step 600 -> 200, more change = lighter). Both ramps pass the ordinal checks of the
dataviz validator (monotone lightness, adjacent gaps >= 0.06, light end >= 2:1 vs surface).
Cells at or below the noise floor or the control are grey and hatched, never blue.
"""

from __future__ import annotations

import math
from collections import Counter
from html import escape
from string import Template

from trajectory_explorer import noise
from trajectory_explorer._version import __version__
from trajectory_explorer.arch import COMPONENTS, classify
from trajectory_explorer.diff import DiffResult, GroupMetrics, tensor_kind
from trajectory_explorer.metrics import TensorMetrics

N_BINS = 5
COMPONENT_LABELS = {
    "embed": "embedding",
    "attn_qkv": "attn QKV",
    "attn_out": "attn out",
    "mlp_in": "MLP in",
    "mlp_out": "MLP out",
    "norm": "layer norm",
    "unembed": "unembed",
    "other": "other",
}
STATUS_TEXT = {
    noise.SIGNIFICANT: "above the noise floor",
    noise.FROM_ZERO: "moved away from exactly zero",
    noise.BELOW_FLOOR: "at or below the rounding noise floor",
    noise.BELOW_CONTROL: "above the floor but at or below the control scale",
    noise.NO_CHANGE: "no change",
}

# Short forms for the table's status column (the full wording is in tooltips and the legend).
STATUS_SHORT = {
    noise.SIGNIFICANT: "above floor",
    noise.FROM_ZERO: "from zero",
    noise.BELOW_FLOOR: "≤ floor",
    noise.BELOW_CONTROL: "≤ control",
    noise.NO_CHANGE: "unchanged",
}

# Heatmap geometry (SVG user units).
_LABEL_W, _HEAD_H, _CELL_W, _CELL_H, _GAP = 120, 28, 92, 34, 2


# -- formatting ---------------------------------------------------------------------------
def pct(value: float | None) -> str:
    """Relative change as a percentage with 3 significant digits."""
    if value is None:
        return "from 0"
    if value == 0.0:
        return "0%"
    percent = value * 100
    if percent >= 1000:  # no scientific notation for big changes: 4,030% rather than 4.03e+03%
        return f"{percent:,.0f}%"
    return f"{percent:.3g}%"


def sci(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2e}"


def _where(layer: int | None, component: str) -> str:
    label = COMPONENT_LABELS.get(component, component)
    return f"layer {layer} {label}" if layer is not None else label


# -- heatmap ------------------------------------------------------------------------------
KIND_PLURAL = {"matrix": "matrices", "vector": "vectors"}


def _row_key(group: GroupMetrics) -> str:
    """Heatmap row: the layer (or embed/final) and the tensor kind, never mixed (D41)."""
    if group.layer is not None:
        base = f"layer {group.layer}"
    else:
        base = "embed" if group.component == "embed" else "final"
    return f"{base} {KIND_PLURAL[group.kind]}"


def _rows(groups: tuple[GroupMetrics, ...]) -> list[str]:
    keys = {_row_key(g) for g in groups}
    layers = sorted(dict.fromkeys(g.layer for g in groups if g.layer is not None))
    bases = ["embed", *[f"layer {n}" for n in layers], "final"]
    return [
        f"{base} {plural}"
        for base in bases
        for plural in ("matrices", "vectors")
        if f"{base} {plural}" in keys
    ]


def _bin_edges(groups: tuple[GroupMetrics, ...]) -> list[float] | None:
    """Log-spaced bin edges spanning the significant groups' relative deltas."""
    values = [g.rel_delta for g in groups if g.status == noise.SIGNIFICANT and g.rel_delta]
    if not values:
        return None
    lo, hi = math.log10(min(values)), math.log10(max(values))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    return [10 ** (lo + (hi - lo) * i / N_BINS) for i in range(N_BINS + 1)]


def _bin(value: float, edges: list[float]) -> int:
    for i in range(N_BINS):
        if value <= edges[i + 1]:
            return i
    return N_BINS - 1


def vector_share_lines(
    g: GroupMetrics, tensors: dict[str, TensorMetrics] | None = None
) -> list[str]:
    """For vector cells: the top-5% share of each tensor (the number replaces the label, D43)."""
    if g.kind != "vector" or not tensors:
        return []
    lines = []
    for name in g.tensors:
        m = tensors.get(name)
        if m is not None and m.top_row_share is not None:
            short = name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1]
            lines.append(
                f"{short}: top 5% of elements hold {pct(m.top_row_share)} of the squared change"
            )
    return lines


def _cell_tooltip(g: GroupMetrics, tensors: dict[str, TensorMetrics] | None = None) -> str:
    lines = [
        f"{_where(g.layer, g.component)} ({g.kind})",
        f"relative change {pct(g.rel_delta)} (||dW||/||W|| = {sci(g.rel_delta)})",
        f"noise floor {sci(g.floor)}",
    ]
    if g.control is not None:
        lines.append(f"control scale {sci(g.control)}")
    lines.append(f"status: {STATUS_TEXT.get(g.status, g.status)}")
    lines.append(f"{len(g.tensors)} tensor(s): " + ", ".join(g.tensors))
    lines.extend(vector_share_lines(g, tensors))
    return "\n".join(lines)


_CELL = Template(
    '<g class="cell $cls"><title>$tip</title>'
    '<rect x="$x" y="$y" width="$w" height="$h" rx="3"$fill/>'
    '<text x="$tx" y="$ty" class="cell-text">$text</text></g>'
)


def render_heatmap(result: DiffResult) -> str:
    groups = result.groups
    rows = _rows(groups)
    present = {g.component for g in groups}
    cols = [c for c in COMPONENTS if c in present]
    edges = _bin_edges(groups)
    by_pos = {(_row_key(g), g.component): g for g in groups}

    width = _LABEL_W + len(cols) * (_CELL_W + _GAP)
    height = _HEAD_H + len(rows) * (_CELL_H + _GAP)
    parts = []
    for j, comp in enumerate(cols):
        x = _LABEL_W + j * (_CELL_W + _GAP) + _CELL_W / 2
        label = escape(COMPONENT_LABELS.get(comp, comp))
        parts.append(f'<text x="{x}" y="18" class="col-label">{label}</text>')
    for i, row in enumerate(rows):
        y = _HEAD_H + i * (_CELL_H + _GAP)
        parts.append(
            f'<text x="{_LABEL_W - 8}" y="{y + _CELL_H / 2 + 4}" class="row-label">'
            f"{escape(row)}</text>"
        )
        for j, comp in enumerate(cols):
            g = by_pos.get((row, comp))
            if g is None:
                continue
            x = _LABEL_W + j * (_CELL_W + _GAP)
            if g.status == noise.SIGNIFICANT and edges and g.rel_delta:
                cls, fill, text = f"sig bin{_bin(g.rel_delta, edges)}", "", pct(g.rel_delta)
            elif g.status == noise.FROM_ZERO:
                cls, fill, text = f"sig bin{N_BINS - 1}", "", "from 0"
            else:
                cls, fill, text = f"muted {g.status}", ' fill="url(#hatch)"', ""
            parts.append(
                _CELL.substitute(
                    cls=cls,
                    tip=escape(_cell_tooltip(g, result.tensors)),
                    x=x,
                    y=y,
                    w=_CELL_W,
                    h=_CELL_H,
                    fill=fill,
                    tx=x + _CELL_W / 2,
                    ty=y + _CELL_H / 2 + 4,
                    text=escape(text),
                )
            )
    n_sig = sum(1 for g in groups if g.significant)
    aria = (
        f"Heatmap of relative change by layer and component: {n_sig} of {len(groups)} cells "
        "above the noise floor."
    )
    return (
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" width="{width}" role="img" '
        f'aria-label="{escape(aria)}">'
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="6" height="6" class="hatch-bg"/>'
        '<line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>'
        + "".join(parts)
        + "</svg>"
        + _legend(edges)
    )


def _legend(edges: list[float] | None) -> str:
    items = []
    if edges:
        for i in range(N_BINS):
            items.append(
                f'<span class="key"><svg width="18" height="14" aria-hidden="true">'
                f'<rect class="cell sig bin{i}" width="18" height="14" rx="3"/></svg>'
                f"{pct(edges[i])} &ndash; {pct(edges[i + 1])}</span>"
            )
    items.append(
        '<span class="key"><svg width="18" height="14" aria-hidden="true"><defs>'
        '<pattern id="hatch-key" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="6" height="6" class="hatch-bg"/>'
        '<line x1="0" y1="0" x2="0" y2="6" class="hatch-line"/></pattern></defs>'
        '<rect width="18" height="14" rx="3" fill="url(#hatch-key)"/></svg>'
        "at or below the noise floor / control, or unchanged</span>"
    )
    scale = "Relative change per cell, log-spaced bins." if edges else ""
    return f'<p class="legend">{scale} {"".join(items)}</p>'


# -- ranked table -------------------------------------------------------------------------
def _sort_key(item: tuple[str, TensorMetrics]) -> tuple[int, float]:
    m = item[1]
    return (0, -m.abs_delta) if m.rel_delta is None else (1, -m.rel_delta)


def _num(value: float) -> str:
    return f"{value:.3g}"


def _name_cells(rank: int, name: str, m: TensorMetrics) -> str:
    key = classify(name)
    breakable = escape(name).replace(".", ".<wbr>")  # break long names at dots only
    shape = escape(" x ".join(map(str, m.shape)))
    return (
        f'<td>{rank}</td><td class="name">{breakable}</td>'
        f'<td>{escape(_where(key.layer, key.component))}</td><td class="nowrap">{shape}</td>'
    )


def _change_cells(m: TensorMetrics) -> str:
    """Relative change, absolute change and the starting norm, so a tiny start is visible."""
    return (
        f'<td class="num">{pct(m.rel_delta)}</td><td class="num">{_num(m.abs_delta)}</td>'
        f'<td class="num">{_num(m.norm_a)}</td>'
    )


_CHANGE_HEADS = (
    '<th class="num">relative change</th><th class="num">absolute change ||&Delta;W||</th>'
    '<th class="num">starting norm ||W<sub>A</sub>||</th>'
)


def _matrix_table(rows: list[tuple[str, TensorMetrics]], table_id: str) -> str:
    body = []
    for rank, (name, m) in enumerate(rows, start=1):
        erank = "&ndash;" if m.erank is None else f"{m.erank:.1f} of {m.rank_dims}"
        body.append(
            f'<tr class="{escape(m.status)}">{_name_cells(rank, name, m)}{_change_cells(m)}'
            f'<td class="num">{erank}</td>'
            f'<td class="num">{"&ndash;" if m.r90 is None else m.r90}</td>'
            f'<td class="nowrap">{escape(m.rank_label)}</td>'
            f'<td class="nowrap">{escape(m.concentration_label)}</td>'
            f'<td class="nowrap">{escape(STATUS_SHORT.get(m.status, m.status))}</td></tr>'
        )
    return (
        f'<table id="{table_id}"><thead><tr><th>#</th><th>tensor</th><th>where</th>'
        f"<th>shape</th>{_CHANGE_HEADS}"
        '<th class="num">effective rank</th><th class="num">r90</th><th>rank</th>'
        "<th>spread</th><th>status</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _vector_table(rows: list[tuple[str, TensorMetrics]], table_id: str) -> str:
    body = []
    for rank, (name, m) in enumerate(rows, start=1):
        share = "&ndash;" if m.top_row_share is None else pct(m.top_row_share)
        body.append(
            f'<tr class="{escape(m.status)}">{_name_cells(rank, name, m)}{_change_cells(m)}'
            f'<td class="num">{share}</td>'
            f'<td class="nowrap">{escape(STATUS_SHORT.get(m.status, m.status))}</td></tr>'
        )
    return (
        f'<table id="{table_id}"><thead><tr><th>#</th><th>tensor</th><th>where</th>'
        f"<th>length</th>{_CHANGE_HEADS}"
        '<th class="num">top 5% of elements hold</th><th>status</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_ranked(result: DiffResult) -> str:
    """Two tables, weight matrices first, then vectors (biases, norm scales), D42."""
    if result.identical:
        return "<p>The files are byte-identical, so no tensor was measured.</p>"
    ordered = sorted(result.tensors.items(), key=_sort_key)
    matrices = [(n, m) for n, m in ordered if tensor_kind(m.shape) == "matrix"]
    vectors = [(n, m) for n, m in ordered if tensor_kind(m.shape) == "vector"]
    html = ""
    for title, kind_rows, table, table_id, none_text in (
        ("Weight matrices", matrices, _matrix_table, "ranked-matrices", "weight matrix"),
        ("Vectors (biases, norm scales)", vectors, _vector_table, "ranked-vectors", "vector"),
    ):
        if not kind_rows:
            continue
        significant = [(n, m) for n, m in kind_rows if m.significant]
        html += f"<h3>{title}</h3>"
        html += (
            table(significant, table_id)
            if significant
            else f"<p>No {none_text} changed by more than the noise floor.</p>"
        )
    full = ""
    if matrices:
        full += "<h3>Weight matrices</h3>" + _matrix_table(matrices, "all-matrices")
    if vectors:
        full += "<h3>Vectors</h3>" + _vector_table(vectors, "all-vectors")
    if full:
        html += (
            f"<details><summary>All {len(ordered)} tensors, including those at or below the "
            f"floor</summary>{full}</details>"
        )
    return html


# -- text sections ------------------------------------------------------------------------
def summary_sentence(result: DiffResult) -> tuple[bool, str]:
    """(anything significant?, one-sentence banner). Never claims more than the data shows."""
    if result.identical:
        return False, ("No significant difference: the two files are byte-identical (same sha256).")
    n = result.n_tensors
    sig = result.significant_tensors
    if not sig:
        statuses = Counter(m.status for m in result.tensors.values())
        largest = max((m.rel_delta or 0.0 for m in result.tensors.values()), default=0.0)
        if statuses[noise.BELOW_CONTROL]:
            return False, (
                f"No significant difference: no tensor changed more than the control scale "
                f"({statuses[noise.BELOW_CONTROL]} of {n} are above the rounding floor but at "
                "or below the control)."
            )
        return False, (
            f"No significant difference: all {n} tensors are at or below the noise floor "
            f"(largest relative change {pct(largest)})."
        )

    # Matrices and vectors are reported separately (D41); within each, prefer the largest
    # real relative change and fall back to "from zero" only if nothing else moved.
    def largest(kind: str) -> str | None:
        candidates = [
            (g.layer, g.component, g.rel_delta)
            for g in result.groups
            if g.significant and g.kind == kind
        ]
        if not candidates:  # significant tensors inside groups that are below floor overall
            candidates = [
                (classify(t).layer, classify(t).component, result.tensors[t].rel_delta)
                for t in sig
                if tensor_kind(result.tensors[t].shape) == kind
            ]
        if not candidates:
            return None
        layer, component, rel = max(candidates, key=lambda c: (c[2] is not None, c[2] or 0.0))
        if rel is None:
            return f"{_where(layer, component)} (moved away from zero)"
        return f"{_where(layer, component)} ({pct(rel)})"

    matrix, vector = largest("matrix"), largest("vector")
    parts = []
    if matrix:
        parts.append(f"The largest weight-matrix change is in {matrix}")
    if vector:
        vec = f"the largest vector change (biases, norm scales) is in {vector}"
        parts.append(vec if matrix else "Only vectors changed: " + vec)
    return True, (
        "; ".join(parts)
        + f". {len(sig)} of {n} tensors changed more than the noise floor"
        + (" and the control scale." if result.control else ".")
    )


def precision_rule_note(n_rule: int) -> str:
    """Plain statement of when the float16-content rule (D40) set a tensor's floor."""
    if not n_rule:
        return ""
    return (
        f"<li>{n_rule} tensor comparison(s): stored as float32, but every value is exactly a "
        "float16 number (in both checkpoints), so the float16 rounding floor applies.</li>"
    )


def render_noise_floor(result: DiffResult) -> str:
    if result.identical:
        floors_html = "<li>Not needed: the files are byte-identical.</li>"
    else:
        pairs = Counter(
            (m.dtype_a, m.dtype_b, m.floor, m.f16_rule) for m in result.tensors.values()
        )
        floors_html = "".join(
            f"<li>{count} tensor(s) stored as {escape(da)} in A and {escape(db)} in B"
            + (" (float16 values only: float16 floor)" if rule else "")
            + f": floor {sci(floor)} ({pct(floor)})</li>"
            for (da, db, floor, rule), count in sorted(pairs.items())
        )
        floors_html += precision_rule_note(sum(m.f16_rule for m in result.tensors.values()))
    statuses = Counter(m.status for m in result.tensors.values())
    counts = (
        f"Of {result.n_tensors} compared tensors: "
        f"{statuses[noise.SIGNIFICANT] + statuses[noise.FROM_ZERO]} above the floor "
        f"({statuses[noise.FROM_ZERO]} of them starting from exactly zero), "
        f"{statuses[noise.BELOW_FLOOR]} at or below the floor, "
        f"{statuses[noise.BELOW_CONTROL]} at or below the control scale, "
        f"{statuses[noise.NO_CHANGE] + (result.n_tensors if result.identical else 0)} unchanged."
    )
    if result.control:
        ca, cb = result.control
        control = (
            f"<p>Control pair: <code>{escape(ca.label)}</code> &rarr; <code>{escape(cb.label)}"
            "</code>. Each tensor's relative change in the control pair is used as a reference "
            "scale, and changes at or below it are muted as &ldquo;below control&rdquo;. A "
            "control such as two adjacent training checkpoints is <strong>not a null</strong>: "
            "the model really learns between them, so it only says how big a change is compared "
            "with one training interval.</p>"
        )
    else:
        control = "<p>No control pair was given (<code>--control</code>).</p>"
    return (
        "<p>The noise floor is the smallest relative change that storage rounding alone can "
        "produce. It is 3 &times; &radic;(u<sub>A</sub>&sup2; + u<sub>B</sub>&sup2;) / &radic;3, "
        "where u is the unit roundoff of each tensor's stored format (float32 2<sup>-24</sup>, "
        "float16 2<sup>-11</sup>, bfloat16 2<sup>-8</sup>). A change at or below it is "
        "consistent with rounding only and is shown muted.</p>"
        f"<ul>{floors_html}</ul>{control}<p>{counts}"
        + (
            f" {len(result.skipped)} non-parameter buffer(s) (masks, rotary tables) were skipped."
            if result.skipped
            else ""
        )
        + "</p>"
    )


LABELS_HTML = """
<dl>
<dt>low-rank</dt><dd>The change uses only a few independent directions: its effective rank is
under a quarter of what random noise of the same shape would have. Typical of fine-tuning-like,
targeted edits.</dd>
<dt>dense</dt><dd>The change uses many directions, like random noise of the same shape would.</dd>
<dt>concentrated</dt><dd>At least half of the change sits in the top 5% of rows (for an
embedding: a small set of tokens). Weight matrices only.</dd>
<dt>spread out</dt><dd>The change is distributed over many rows. Weight matrices only; for
vectors (biases, norm scales) the report shows the share of the change held by the top 5% of
elements as a number instead of a label.</dd>
<dt>from zero</dt><dd>The tensor was exactly zero in A (for example a bias at initialisation),
so a relative change is undefined; any non-zero value in B counts as a real change.</dd>
<dt>effective rank, r90</dt><dd>Effective rank is exp(entropy) of the normalised singular
values: k equal directions give exactly k. r90 is how many directions hold 90% of the
change.</dd>
</dl>
<p>The rank and spread labels answer different questions, so both can be true at once: a change
confined to 200 rows of an embedding is <em>concentrated</em> (few rows) and also <em>dense</em>
(those 200 rows point in about 200 different directions).</p>
"""


# -- page ---------------------------------------------------------------------------------
BASE_CSS = """:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hair: #e1e0d9; --border: rgba(11,11,11,0.10);
  --b0: #86b6ef; --b1: #5598e7; --b2: #2a78d6; --b3: #1c5cab; --b4: #104281;
  --t0: #0b0b0b; --t1: #0b0b0b; --t2: #ffffff; --t3: #ffffff; --t4: #ffffff;
  --hatch-bg: #f0efec; --hatch-line: #c3c2b7;
  --banner-ok: #eef5ee; --banner-sig: #eaf2fd;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --hair: #2c2c2a; --border: rgba(255,255,255,0.10);
    --b0: #184f95; --b1: #256abf; --b2: #3987e5; --b3: #6da7ec; --b4: #9ec5f4;
    --t0: #ffffff; --t1: #ffffff; --t2: #0b0b0b; --t3: #0b0b0b; --t4: #0b0b0b;
    --hatch-bg: #383835; --hatch-line: #52514e;
    --banner-ok: #1d261d; --banner-sig: #16233a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 4px 20px 16px; margin-top: 16px; }
.meta { color: var(--ink-2); font-size: 13px; margin: 0; }
.meta code { font-size: 12px; }
.banner { border-radius: 8px; padding: 14px 18px; font-size: 17px; font-weight: 600;
  border: 1px solid var(--border); margin-top: 16px; }
.banner.none { background: var(--banner-ok); }
.banner.some { background: var(--banner-sig); }
.scroll { overflow-x: auto; }
svg.heatmap { max-width: 100%; height: auto; font-size: 12px; }
.col-label { text-anchor: middle; fill: var(--ink-2); }
.row-label { text-anchor: end; fill: var(--ink-2); }
.cell-text { text-anchor: middle; font-size: 11px; font-variant-numeric: tabular-nums; }
.cell.bin0 rect, rect.bin0 { fill: var(--b0); } .cell.bin0 .cell-text { fill: var(--t0); }
.cell.bin1 rect, rect.bin1 { fill: var(--b1); } .cell.bin1 .cell-text { fill: var(--t1); }
.cell.bin2 rect, rect.bin2 { fill: var(--b2); } .cell.bin2 .cell-text { fill: var(--t2); }
.cell.bin3 rect, rect.bin3 { fill: var(--b3); } .cell.bin3 .cell-text { fill: var(--t3); }
.cell.bin4 rect, rect.bin4 { fill: var(--b4); } .cell.bin4 .cell-text { fill: var(--t4); }
.hatch-bg { fill: var(--hatch-bg); } .hatch-line { stroke: var(--hatch-line); stroke-width: 2; }
.legend { color: var(--ink-2); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px 16px;
  align-items: center; }
.key { display: inline-flex; align-items: center; gap: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--hair);
  vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.name { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
td.nowrap { white-space: nowrap; }
tr.below_floor td, tr.below_control td, tr.no_change td { color: var(--muted); }
details { margin-top: 12px; } summary { cursor: pointer; color: var(--ink-2); }
dt { font-weight: 600; margin-top: 8px; } dd { margin: 0 0 0 16px; color: var(--ink-2); }
h3 { font-size: 14px; margin: 14px 0 6px; }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
@media print {
  /* Always print the light palette, even when the viewer's OS is in dark mode. */
  :root {
    color-scheme: light;
    --surface: #ffffff; --page: #ffffff; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --hair: #e1e0d9; --border: #c3c2b7;
    --b0: #86b6ef; --b1: #5598e7; --b2: #2a78d6; --b3: #1c5cab; --b4: #104281;
    --t0: #0b0b0b; --t1: #0b0b0b; --t2: #ffffff; --t3: #ffffff; --t4: #ffffff;
    --hatch-bg: #f0efec; --hatch-line: #c3c2b7;
    --banner-ok: #eef5ee; --banner-sig: #eaf2fd;
  }
  body { font-size: 10pt; }
  main { max-width: none; padding: 0; }
  section { padding: 2px 10px 8px; }
  #summary, #heatmap, #noise-floor, #labels { break-inside: avoid; }
  tr { break-inside: avoid; }
  table { font-size: 8pt; }
  td.name { font-size: 7.5pt; }
  th, td { padding: 3px 4px; }
  svg.heatmap, .legend svg, .banner {
    print-color-adjust: exact; -webkit-print-color-adjust: exact;
  }
  .scroll { overflow: visible; }
}
"""

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
<p class="meta">A: <code>$a_label</code> ($a_path, sha256 $a_sha, $a_size) &middot;
B: <code>$b_label</code> ($b_path, sha256 $b_sha, $b_size)</p>
<div id="summary" class="banner $banner_cls" role="status">$banner</div>

<section id="heatmap">
<h2>Where the model changed: layer &times; component</h2>
<p class="meta">Relative change ||&Delta;W|| / ||W|| per cell, aggregated over its tensors by
sums of squares. Hover a cell for exact numbers.</p>
<div class="scroll">$heatmap</div>
</section>

<section id="ranked">
<h2>What moved most</h2>
<p class="meta">Tensors above the noise floor, largest relative change first, weight matrices
and vectors in separate tables. The starting norm shows when a big relative change comes from a
tiny starting size.</p>
<div class="scroll">$ranked</div>
</section>

<section id="noise-floor">
<h2>Noise floor</h2>
$noise
</section>

<section id="labels">
<h2>What the labels mean</h2>
$labels
</section>

<footer>Generated by trajectory-explorer $version (metrics v$metrics_version, schema
v$schema_version). Weight-only comparison: no inputs were run through either model.</footer>
</main>
</body>
</html>
""")


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n:,} B"


def render_html(result: DiffResult) -> str:
    """Render the full report as one HTML string."""
    any_sig, banner = summary_sentence(result)
    return _PAGE.substitute(
        version=escape(__version__),
        title=escape(f"{result.a.label} vs {result.b.label}"),
        a_label=escape(result.a.label),
        a_path=escape(result.a.path),
        a_sha=escape(result.a.sha256[:12]),
        a_size=_size(result.a.size_bytes),
        b_label=escape(result.b.label),
        b_path=escape(result.b.path),
        b_sha=escape(result.b.sha256[:12]),
        b_size=_size(result.b.size_bytes),
        banner_cls="some" if any_sig else "none",
        banner=escape(banner),
        heatmap=render_heatmap(result),
        ranked=render_ranked(result),
        noise=render_noise_floor(result),
        labels=LABELS_HTML,
        css=BASE_CSS,
        metrics_version=result.metrics_version,
        schema_version=result.schema_version,
    )

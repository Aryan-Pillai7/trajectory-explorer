"""From-zero cells have their own look and never sit on the colour scale (D50)."""

import re

import numpy as np
import pytest

from trajectory_explorer.diff import diff_checkpoints, local_source
from trajectory_explorer.report import render_heatmap
from trajectory_explorer.trajectory import run_trajectory
from trajectory_explorer.trajectory_report import render_heatmap as render_trajectory_heatmap

W = "gpt_neox.layers.0.attention.query_key_value.weight"
B = "gpt_neox.layers.0.attention.query_key_value.bias"


@pytest.fixture
def bias_from_zero(make_checkpoint):
    a = {W: np.ones((24, 8), np.float32), B: np.zeros(24, np.float32)}
    b = {W: np.full((24, 8), 1.1, np.float32), B: np.full(24, 0.5, np.float32)}
    return make_checkpoint("a.safetensors", a), make_checkpoint("b.safetensors", b)


def vector_panel(html: str, svg_id: str) -> str:
    start = html.index(f'id="{svg_id}"')
    return html[start : html.index("</svg>", start)]


@pytest.mark.integration
@pytest.mark.parametrize("report", ["pair", "trajectory"])
def test_from_zero_cells_are_marked_and_off_the_colour_scale(report, bias_from_zero):
    if report == "pair":
        html = render_heatmap(diff_checkpoints(*bias_from_zero))
        panel = vector_panel(html, "heatmap-vector")
    else:
        result = run_trajectory([local_source(p) for p in bias_from_zero])
        html = render_trajectory_heatmap(result)
        panel = vector_panel(html, "t-heatmap-vector")

    cells = re.findall(r'<g class="cell ([^"]+)">', panel)
    assert cells == ["fromzero"]  # not "sig bin4", not hatched
    assert "→</text>" in panel  # "0->" is drawn on the cell itself
    assert "moved away from exactly zero" in panel  # the exact reason stays in the tooltip
    # The legend lists three categories: the scale, from zero, and at/below the floor.
    assert "moved away from exactly zero (no relative change exists)" in html
    assert "at or below the noise floor" in html

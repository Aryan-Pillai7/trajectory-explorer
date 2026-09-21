"""Matrices and vectors get separate colour scales (D46)."""

import re

import numpy as np
import pytest

from trajectory_explorer.diff import diff_checkpoints, local_source
from trajectory_explorer.report import render_heatmap
from trajectory_explorer.trajectory import run_trajectory
from trajectory_explorer.trajectory_report import render_heatmap as render_trajectory_heatmap

W = "gpt_neox.layers.5.attention.query_key_value.weight"
B = "gpt_neox.layers.5.attention.query_key_value.bias"


@pytest.fixture
def weight_and_tiny_bias(make_checkpoint, tmp_path):
    """A weight that moves 10% and a tiny bias that grows 300x (the real Pythia shape)."""
    a = {W: np.ones((24, 8), np.float32), B: np.full(24, 0.01, np.float32)}
    b = {W: np.full((24, 8), 1.1, np.float32), B: np.full(24, 3.0, np.float32)}
    return make_checkpoint("a.safetensors", a), make_checkpoint("b.safetensors", b)


def panel(html: str, svg_id: str) -> str:
    """The SVG of one panel plus the legend right after it."""
    start = html.index(f'id="{svg_id}"')
    end = html.index("</p>", html.index("</svg>", start))
    return html[start:end]


def cell_bin(svg: str) -> int:
    (found,) = re.findall(r'<g class="cell sig bin(\d)"', svg)  # cells, not legend keys
    return int(found)


@pytest.mark.integration
def test_a_huge_vector_change_does_not_set_the_matrix_scale(weight_and_tiny_bias):
    html = render_heatmap(diff_checkpoints(*weight_and_tiny_bias))
    matrices, vectors = panel(html, "heatmap-matrix"), panel(html, "heatmap-vector")

    assert "29,900%" not in matrices  # the bias's number is not on the matrix scale
    assert "29,900%" in vectors
    # Under one shared scale the 10% weight would sit in the lightest bin; on its own it is central.
    assert cell_bin(matrices) == 2 and cell_bin(vectors) == 2
    assert "colours are not comparable between the matrices panel and the vectors panel" in html
    assert (
        "Weight matrices scale:" in matrices and "Vectors (biases, norm scales) scale:" in vectors
    )


@pytest.mark.integration
def test_trajectory_panels_have_their_own_scales(weight_and_tiny_bias):
    result = run_trajectory([local_source(p) for p in weight_and_tiny_bias])
    html = render_trajectory_heatmap(result)
    matrices, vectors = panel(html, "t-heatmap-matrix"), panel(html, "t-heatmap-vector")
    assert "29,900%" not in matrices and cell_bin(matrices) == 2
    assert cell_bin(vectors) == 2
    assert "colours are not comparable" in html

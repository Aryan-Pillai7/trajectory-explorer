"""Trajectory report layout: fits at 1280 px up to 42 intervals, scrolls beyond (D49)."""

import re
from dataclasses import replace

import numpy as np
import pytest

from trajectory_explorer.diff import diff_checkpoints
from trajectory_explorer.trajectory import TrajectoryResult
from trajectory_explorer.trajectory_report import heatmap_geometry, render_heatmap

W = "gpt_neox.layers.0.attention.query_key_value.weight"


def fake_trajectory(interval, n):
    points = tuple(replace(interval.a, label=f"x@step{i}") for i in range(n + 1))
    return TrajectoryResult(points=points, intervals=(interval,) * n)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n", "fits"), [(1, True), (24, True), (30, True), (42, True), (43, False)]
)
def test_heatmap_geometry_fits_the_width_budget_until_cells_get_too_small(n, fits):
    cell, width, scrolls = heatmap_geometry(n)
    assert scrolls is not fits
    if fits:
        assert width <= 1000 + 1e-9 and cell >= 16
    else:
        assert cell == 16 and width > 1000


@pytest.mark.integration
@pytest.mark.parametrize(("n", "scrolls"), [(30, False), (50, True)])
def test_panels_scale_to_the_page_or_scroll_past_the_limit(n, scrolls, make_checkpoint):
    a = make_checkpoint("a.safetensors", {W: np.ones((24, 8), np.float32)})
    b = make_checkpoint("b.safetensors", {W: np.full((24, 8), 1.1, np.float32)})
    html = render_heatmap(fake_trajectory(diff_checkpoints(a, b), n))
    (svg,) = re.findall(r"<svg class=\"traj-heatmap\"[^>]*>", html)
    view_w = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
    if scrolls:
        assert f"min-width: {view_w:.0f}px" in svg  # the .scroll box scrolls sideways
    else:
        assert 'width="100%"' in svg and view_w <= 1000  # scales into the section
    assert html.count('<text class="col-head"') == n  # every interval has a labelled column

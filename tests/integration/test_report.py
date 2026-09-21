import shutil

import numpy as np
import pytest

from conftest import neox_tensors, parse_report, perturb
from trajectory_explorer.diff import diff_checkpoints
from trajectory_explorer.report import render_html


@pytest.mark.integration
def test_report_has_sections_in_order_is_offline_and_ranks_changes(make_checkpoint, rng):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", perturb(base, 0.01, rng))
    html = render_html(diff_checkpoints(a, b))
    report = parse_report(html)

    assert html.startswith("<!DOCTYPE html>")
    assert report.section_ids == ["heatmap", "ranked", "noise-floor", "labels"]
    assert html.index('id="summary"') < html.index('id="heatmap"')
    assert report.banner.startswith("The largest weight-matrix change is in")
    assert report.sig_cells > 0
    assert report.ranked_rows == len(base)  # every tensor moved 1%: all significant
    # Two tables: weight matrices first, then vectors; rank labels only for matrices.
    n_matrices = sum(t.ndim >= 2 for t in base.values())
    assert report.table_rows["ranked-matrices"] == n_matrices
    assert report.table_rows["ranked-vectors"] == len(base) - n_matrices
    assert html.index('id="ranked-matrices"') < html.index('id="ranked-vectors"')
    for table in ("ranked-matrices", "ranked-vectors"):
        heads = report.table_heads[table]
        assert "starting norm ||WA||" in heads and "absolute change ||ΔW||" in heads
    assert "rank, spread" in report.table_heads["ranked-matrices"]
    assert not any("rank" in h for h in report.table_heads["ranked-vectors"])
    assert len(report.table_heads["ranked-matrices"]) <= 9  # fits a laptop screen
    assert "status" in report.table_heads["all-matrices"]  # the full lists keep the status
    assert report.table_rows["all-matrices"] + report.table_rows["all-vectors"] == len(base)
    assert report.external_refs == []
    assert report.forbidden_tags == []


@pytest.mark.unit
def test_percentages_never_use_scientific_notation_for_big_changes():
    from trajectory_explorer.report import pct

    assert pct(40.308) == "4,031%"  # the real pythia step1000 -> step143000 QKV cell
    assert pct(283.26) == "28,326%"
    assert pct(0.0123) == "1.23%"
    assert pct(None) == "from 0"


@pytest.mark.integration
@pytest.mark.parametrize("null_pair", ["byte_identical", "float16_rounding_only"])
def test_null_pairs_render_no_significant_difference(null_pair, make_checkpoint, rng, tmp_path):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    if null_pair == "byte_identical":
        b = tmp_path / "b.safetensors"
        shutil.copyfile(a, b)
    else:
        b = make_checkpoint("b.safetensors", {n: t.astype(np.float16) for n, t in base.items()})

    report = parse_report(render_html(diff_checkpoints(a, b)))

    assert report.banner.startswith("No significant difference")
    assert report.sig_cells == 0
    assert report.ranked_rows == 0
    assert report.section_ids == ["heatmap", "ranked", "noise-floor", "labels"]

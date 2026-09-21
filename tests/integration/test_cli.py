import json

import numpy as np
import pytest

from conftest import neox_tensors, parse_report, perturb
from trajectory_explorer import __version__, noise
from trajectory_explorer.cli import main


def run(argv):
    """Call the CLI like the shell would: return the exit code, also for argparse exits."""
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code


@pytest.mark.integration
def test_diff_end_to_end_writes_html_json_and_default_path(
    make_checkpoint, rng, tmp_path, isolated_data_dir, capsys
):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b_dir = tmp_path / "step2000"
    b_dir.mkdir()
    b = make_checkpoint("step2000/model.safetensors", perturb(base, 0.01, rng))
    assert b.parent == b_dir

    out, js = tmp_path / "report.html", tmp_path / "result.json"
    assert run(["diff", str(a), str(b_dir), "-o", str(out), "--json", str(js)]) == 0
    report = parse_report(out.read_text(encoding="utf-8"))
    assert report.section_ids == ["heatmap", "ranked", "noise-floor", "labels"]
    assert report.ranked_rows == len(base)
    result = json.loads(js.read_text(encoding="utf-8"))
    assert result["b"]["label"] == "step2000"  # directory name labels model.safetensors
    printed = capsys.readouterr().out
    assert f"Report: {out}" in printed and "The largest change is in" in printed

    # Without -o the report goes to $TE_DATA_DIR/reports, and the metrics cache is used.
    assert run(["diff", str(a), str(b)]) == 0
    reports = list((isolated_data_dir / "reports").glob("diff_a_vs_step2000_*.html"))
    assert len(reports) == 1
    assert str(reports[0]) in capsys.readouterr().out
    assert len(list((isolated_data_dir / "metrics").glob("*.json"))) == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "code", "message"),
    [
        ("version", 0, f"trajectory-explorer {__version__}"),
        ("bad_args", 2, "the following arguments are required"),
        ("bad_control", 2, "expected A2:B2"),
        ("mismatch", 3, "gpt_neox.embed_in.weight: A[40, 8] vs B[40, 16]"),
        ("missing", 4, "Checkpoint not found"),
        ("hub_source", 4, "Hub sources are not built yet"),
    ],
)
def test_exit_codes(case, code, message, make_checkpoint, rng, tmp_path, capsys):
    a = make_checkpoint("a.safetensors", neox_tensors(rng))
    argv = {
        "version": ["--version"],
        "bad_args": ["diff", str(a)],
        "bad_control": ["diff", str(a), str(a), "--control", "nocolon"],
        "mismatch": [
            "diff",
            str(a),
            str(make_checkpoint("wide.safetensors", neox_tensors(rng, hidden=16))),
        ],
        "missing": ["diff", str(a), str(tmp_path / "nope.safetensors")],
        "hub_source": ["diff", "EleutherAI/pythia-70m@step1000", str(a)],
    }[case]
    assert run(argv) == code
    captured = capsys.readouterr()
    assert message in captured.out + captured.err


@pytest.mark.integration
def test_diff_with_control_mutes_changes_below_the_control(make_checkpoint, rng, tmp_path):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", perturb(base, 0.01, rng))
    big = make_checkpoint("c.safetensors", perturb(base, 0.05, rng))
    js, out = tmp_path / "r.json", tmp_path / "r.html"

    assert (
        run(["diff", str(a), str(b), "--control", f"{a}:{big}", "-o", str(out), "--json", str(js)])
        == 0
    )

    result = json.loads(js.read_text(encoding="utf-8"))
    assert {m["status"] for m in result["tensors"].values()} == {noise.BELOW_CONTROL}
    assert result["control"][1]["label"] == "c"
    html = out.read_text(encoding="utf-8")
    assert parse_report(html).banner.startswith("No significant difference")
    assert "not a null" in html
    np.testing.assert_allclose([m["control"] for m in result["tensors"].values()], 0.05, rtol=1e-3)

import json

import numpy as np
import pytest

from conftest import checkpoint_bytes, neox_tensors, parse_report, perturb, write_safetensors
from trajectory_explorer import cli, hub, metrics
from trajectory_explorer.cli import main
from trajectory_explorer.trajectory import TrajectoryResult, select_steps

REPO = "EleutherAI/pythia-tiny"
QKV0 = "gpt_neox.layers.0.attention.query_key_value.weight"
MLP_OUT1 = "gpt_neox.layers.1.mlp.dense_4h_to_h.weight"


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code


def step_dirs(tmp_path, checkpoints):
    """Write {step: tensors} as <tmp>/steps/stepN/model.safetensors; return the dirs in order."""
    dirs = []
    for step, tensors in checkpoints.items():
        folder = tmp_path / "steps" / f"step{step}"
        folder.mkdir(parents=True)
        write_safetensors(folder / "model.safetensors", tensors)
        dirs.append(str(folder))
    return dirs


def significant_cells(result_json):
    return [
        {(g["layer"], g["component"]) for g in r["groups"] if g["status"] == "significant"}
        for r in result_json["intervals"]
    ]


@pytest.mark.unit
def test_default_step_selection_is_log_spaced_and_snapped():
    pythia = [0, *[2**k for k in range(10)], *range(1000, 143001, 1000)]  # 154 real steps
    chosen = select_steps(pythia)
    assert 25 <= len(chosen) <= 26
    assert chosen[0] == 0 and chosen[-1] == 143000
    assert chosen == sorted(set(chosen)) and set(chosen) <= set(pythia)
    assert chosen[:10] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]  # dense where log-spacing is
    assert select_steps([0, 10, 20]) == [0, 10, 20]  # fewer than 25: all of them


@pytest.mark.integration
def test_planted_changes_show_up_in_the_right_interval(tmp_path, rng):
    s0 = neox_tensors(rng)
    s1 = {**s0, QKV0: perturb({QKV0: s0[QKV0]}, 0.05, rng)[QKV0]}
    s2 = {**s1, MLP_OUT1: perturb({MLP_OUT1: s1[MLP_OUT1]}, 0.05, rng)[MLP_OUT1]}
    dirs = step_dirs(tmp_path, {0: s0, 1: s1, 2: s2})
    out, js = tmp_path / "t.html", tmp_path / "t.json"

    assert run(["trajectory", *dirs, "-o", str(out), "--json", str(js)]) == 0

    text = js.read_text(encoding="utf-8")
    assert TrajectoryResult.from_json(text).to_json() == text  # exact round trip
    assert significant_cells(json.loads(text)) == [{(0, "attn_qkv")}, {(1, "mlp_out")}]
    html = out.read_text(encoding="utf-8")
    report = parse_report(html)
    assert report.section_ids == [
        "heatmap",
        "lines",
        "intervals",
        "noise-floor",
        "interval-lengths",
    ]
    assert report.banner.startswith(("attn QKV (matrix) moved most", "MLP out (matrix) moved most"))
    assert report.sig_cells == 2
    assert "0 to 1" in html and "1 to 2" in html
    # Matrices and vectors are separate rows, and the line chart has one "all vectors" series.
    assert "L0 attn QKV (matrix)" in html and "L0 attn QKV (vector)" in html
    assert html.count('<span class="key"><svg width="34"') == 7  # 6 matrix components + vectors
    assert "all vectors</span>" in html
    assert html.count('class="end-label"') == 2  # only the two planted series have points
    assert "reference scale, not a null" in html
    # The short version sits right under the banner, before the heatmap (first screen, D51).
    note = html.index('id="reference-note"')
    assert html.index('id="summary"') < note < html.index('id="heatmap"')
    assert "Adjacent-step diffs are a reference scale, not\na null." in html[note : note + 200]
    assert report.external_refs == [] and report.forbidden_tags == []


@pytest.mark.integration
def test_all_null_trajectory_says_no_significant_change(tmp_path, rng):
    s0 = neox_tensors(rng)
    f16 = {n: t.astype(np.float16) for n, t in s0.items()}  # rounding only
    back = {n: t.astype(np.float32) for n, t in f16.items()}  # same values as f16, stored F32
    dirs = step_dirs(tmp_path, {0: s0, 1: f16, 2: back})
    out = tmp_path / "t.html"
    assert run(["trajectory", *dirs, "-o", str(out)]) == 0
    report = parse_report(out.read_text(encoding="utf-8"))
    assert report.banner.startswith("No significant change across the sampled steps")
    assert report.sig_cells == 0


@pytest.fixture
def hub_steps(fake_hub, rng, tmp_path):
    base = neox_tensors(rng)
    for k, step in enumerate((0, 1, 2, 4, 8)):
        tensors = perturb(base, 0.01 * k, rng) if k else base
        fake_hub.add(REPO, f"step{step}", checkpoint_bytes(tmp_path, tensors))
    return fake_hub


@pytest.mark.integration
def test_hub_trajectory_downloads_each_checkpoint_once_within_two_files(
    hub_steps, isolated_data_dir, monkeypatch, capsys
):
    on_disk = []
    real = hub.download

    def recording(remote, dest, **kwargs):
        stats = real(remote, dest, **kwargs)
        on_disk.append(len(list((isolated_data_dir / "checkpoints").rglob("model.safetensors"))))
        return stats

    monkeypatch.setattr(hub, "download", recording)
    assert run(["trajectory", REPO, "--steps", "all"]) == 0
    err = capsys.readouterr().err
    assert "Steps (5 of 5 step branches): 0, 1, 2, 4, 8" in err
    assert "Downloads needed: 5 file(s)" in err
    assert hub_steps.cdn_gets() == ["step0", "step1", "step2", "step4", "step8"]  # each once
    # Right after a download the old file is still there (eviction waits for success, D52),
    # so three are seen at that moment; after the run the store is back to two.
    assert max(on_disk) == 3
    assert len(list((isolated_data_dir / "checkpoints").rglob("model.safetensors"))) == 2

    # Everything is cached now: a rerun downloads and measures nothing.
    assert run(["trajectory", REPO, "--steps", "all"]) == 0
    assert hub_steps.cdn_gets() == ["step0", "step1", "step2", "step4", "step8"]
    assert "Downloads needed" not in capsys.readouterr().err


@pytest.mark.integration
def test_trajectory_guard_over_the_limit_without_a_tty(hub_steps, monkeypatch, capsys):
    monkeypatch.setattr(cli, "GUARD_BYTES", 1000)
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: False)
    assert run(["trajectory", REPO, "--steps", "0,1,2"]) == 4
    err = capsys.readouterr().err
    assert "Downloads needed: 3 file(s)" in err and "Re-run with --yes" in err
    assert hub_steps.cdn_gets() == []


@pytest.mark.integration
def test_cached_intervals_are_not_measured_again(tmp_path, rng, monkeypatch):
    base = neox_tensors(rng)
    points = {step: perturb(base, 0.01 * step, rng) if step else base for step in (0, 1, 2, 3)}
    dirs = step_dirs(tmp_path, points)
    assert run(["diff", dirs[0], dirs[1]]) == 0  # fills the pair cache for two intervals
    assert run(["diff", dirs[1], dirs[2]]) == 0

    calls = {"n": 0}
    real = metrics.measure_tensor

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "measure_tensor", counting)
    assert run(["trajectory", *dirs, "-o", str(tmp_path / "t.html")]) == 0
    assert calls["n"] == len(base)  # only the third interval was measured


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "code", "message"),
    [
        ("no_step_branches", 4, "Pass an explicit ordered list"),
        ("unknown_step", 4, "no branch for step(s) 3"),
        ("single_spec", 2, "needs either org/name"),
        ("steps_with_list", 2, "--steps only applies"),
        ("bad_steps", 2, "expected default, all or a list"),
        ("mismatch", 3, "do not have the same architecture"),
    ],
)
def test_trajectory_exit_codes(case, code, message, hub_steps, tmp_path, rng, capsys):
    hub_steps.add("EleutherAI/no-steps", "main", b"x")
    dirs = step_dirs(tmp_path, {0: neox_tensors(rng), 1: neox_tensors(rng, hidden=16)})
    argv = {
        "no_step_branches": ["trajectory", "EleutherAI/no-steps"],
        "unknown_step": ["trajectory", REPO, "--steps", "0,3"],
        "single_spec": ["trajectory", dirs[0]],
        "steps_with_list": ["trajectory", dirs[0], dirs[1], "--steps", "all"],
        "bad_steps": ["trajectory", REPO, "--steps", "abc"],
        "mismatch": ["trajectory", *dirs],
    }[case]
    assert run(argv) == code
    captured = capsys.readouterr()
    assert message in captured.out + captured.err

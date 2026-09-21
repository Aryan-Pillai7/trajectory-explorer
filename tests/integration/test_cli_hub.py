import json

import pytest

from conftest import checkpoint_bytes, neox_tensors, perturb
from trajectory_explorer import cli, metrics, noise
from trajectory_explorer.cli import main

REPO = "EleutherAI/pythia-tiny"


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code


@pytest.fixture
def four_revisions(fake_hub, rng, tmp_path):
    base = neox_tensors(rng)
    scales = {"s1": 0.0, "s2": 0.01, "s3": 0.05, "s4": 0.10}  # s1 is the base itself
    for rev, scale in scales.items():
        tensors = perturb(base, scale, rng) if scale else base
        fake_hub.add(REPO, rev, checkpoint_bytes(tmp_path, tensors))
    return fake_hub


def stored(data_dir):
    return sorted(p.parent.name for p in (data_dir / "checkpoints").rglob("model.safetensors"))


@pytest.mark.integration
def test_hub_diff_with_control_keeps_two_files_and_finishes_the_main_pair_first(
    four_revisions, isolated_data_dir, tmp_path, capsys
):
    js = tmp_path / "r.json"
    argv = ["diff", f"{REPO}@s1", f"{REPO}@s2", "--control", f"{REPO}@s3:{REPO}@s4"]
    assert run([*argv, "--json", str(js), "-o", str(tmp_path / "r.html")]) == 0

    # Main pair downloaded (and measured) first, then the control pair evicted it.
    assert four_revisions.cdn_gets() == ["s1", "s2", "s3", "s4"]
    assert stored(isolated_data_dir) == ["s3", "s4"]
    result = json.loads(js.read_text(encoding="utf-8"))
    assert result["a"]["path"] == f"{REPO}@s1"  # the report shows the source, not a path
    assert result["b"]["label"] == "pythia-tiny@s2"
    assert {m["status"] for m in result["tensors"].values()} == {noise.BELOW_CONTROL}
    assert "Downloads needed: 4 file(s)" in capsys.readouterr().err

    # Both pairs are now in the metrics cache: nothing is downloaded or re-read, although
    # s1 and s2 were evicted.
    assert run(argv) == 0
    assert four_revisions.cdn_gets() == ["s1", "s2", "s3", "s4"]
    assert "Downloads needed" not in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "code"), [("no_tty", 4), ("yes_flag", 0), ("tty_declines", 4), ("tty_accepts", 0)]
)
def test_download_guard(mode, code, four_revisions, monkeypatch, capsys):
    monkeypatch.setattr(cli, "GUARD_BYTES", 1000)  # the tiny test files count as "large"
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: mode.startswith("tty"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y" if mode == "tty_accepts" else "")
    argv = ["diff", f"{REPO}@s1", f"{REPO}@s2"] + (["--yes"] if mode == "yes_flag" else [])

    assert run(argv) == code
    err = capsys.readouterr().err
    assert "Downloads needed: 2 file(s)" in err
    if mode == "no_tty":
        assert "Re-run with --yes" in err
        assert four_revisions.cdn_gets() == []  # nothing was downloaded
    else:
        assert "Measured" in err and "should take about" in err  # throughput + estimate
        expected = ["s1"] if mode == "tty_declines" else ["s1", "s2"]
        assert four_revisions.cdn_gets() == expected


@pytest.mark.integration
def test_hub_and_local_runs_share_the_metrics_cache(
    fake_hub, make_checkpoint, rng, isolated_data_dir, monkeypatch
):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", perturb(base, 0.01, rng))
    fake_hub.add(REPO, "a", a.read_bytes())
    fake_hub.add(REPO, "b", b.read_bytes())
    assert run(["diff", str(a), str(b)]) == 0  # local run fills the cache

    calls = {"n": 0}
    real = metrics.measure_tensor

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "measure_tensor", counting)
    assert run(["diff", f"{REPO}@a", f"{REPO}@b"]) == 0
    assert calls["n"] == 0  # same LFS sha256 key: cache hit
    assert fake_hub.cdn_gets() == []  # and nothing was downloaded
    assert len(list((isolated_data_dir / "metrics").glob("*.json"))) == 1


@pytest.mark.integration
def test_verification_failure_exits_4_and_leaves_no_file(four_revisions, isolated_data_dir, capsys):
    four_revisions.corrupt.add((REPO, "s2"))
    assert run(["diff", f"{REPO}@s1", f"{REPO}@s2"]) == 4
    assert "failed verification" in capsys.readouterr().err
    assert stored(isolated_data_dir) == ["s1"]
    assert not list((isolated_data_dir / "checkpoints").rglob("*.part"))

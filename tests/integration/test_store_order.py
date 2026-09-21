"""The store must never evict a file the current pair still needs (and the guard must count
what will really be downloaded)."""

import pytest

from conftest import checkpoint_bytes, neox_tensors, perturb
from trajectory_explorer import cli
from trajectory_explorer.cli import main

REPO = "EleutherAI/pythia-tiny"


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code


def stored(data_dir):
    return sorted(p.parent.name for p in (data_dir / "checkpoints").rglob("model.safetensors"))


@pytest.fixture
def steps(fake_hub, rng, tmp_path):
    base = neox_tensors(rng)
    for k, step in enumerate((0, 1, 2, 4, 141000, 142000, 143000)):
        fake_hub.add(REPO, f"step{step}", checkpoint_bytes(tmp_path, perturb(base, 0.01 * k, rng)))
    return fake_hub


@pytest.mark.integration
def test_fetching_a_never_evicts_b_when_b_is_already_stored(steps, isolated_data_dir):
    assert run(["diff", f"{REPO}@step142000", f"{REPO}@step143000"]) == 0
    assert stored(isolated_data_dir) == ["step142000", "step143000"]  # step142000 is the LRU

    # B (step142000) is a store hit; fetching A (step141000) must not evict it.
    assert run(["diff", f"{REPO}@step141000", f"{REPO}@step142000"]) == 0
    assert steps.cdn_gets() == ["step142000", "step143000", "step141000"]  # no re-download
    assert stored(isolated_data_dir) == ["step141000", "step142000"]


@pytest.mark.integration
def test_trajectory_guard_counts_files_that_will_be_evicted_before_their_turn(
    steps, isolated_data_dir, monkeypatch, capsys
):
    # Leave step0 and step4 in the store (step4 newer). The trajectory 0, 1, 2, 4 evicts step4
    # while measuring the early intervals, so step4 has to be downloaded again at the end.
    assert run(["diff", f"{REPO}@step0", f"{REPO}@step4"]) == 0
    before = len(steps.cdn_gets())
    capsys.readouterr()

    monkeypatch.setattr(cli, "GUARD_BYTES", 10**12)  # plan line only, no confirmation needed
    assert run(["trajectory", REPO, "--steps", "0,1,2,4"]) == 0
    planned = capsys.readouterr().err
    actual = steps.cdn_gets()[before:]
    assert actual == ["step1", "step2", "step4"]
    assert f"Downloads needed: {len(actual)} file(s)" in planned  # plan == reality

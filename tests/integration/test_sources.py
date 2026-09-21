import logging
from pathlib import Path

import pytest

from conftest import checkpoint_bytes, neox_tensors, perturb
from trajectory_explorer import hub
from trajectory_explorer.errors import InputError
from trajectory_explorer.hub import HubSpec, download, fetch_metadata
from trajectory_explorer.sources import parse_spec
from trajectory_explorer.store import CheckpointStore

REPO = "EleutherAI/pythia-tiny"


@pytest.mark.unit
def test_spec_parsing(tmp_path, monkeypatch):
    assert parse_spec("EleutherAI/pythia-70m@step143000") == HubSpec(
        "EleutherAI/pythia-70m", "step143000"
    )
    # An existing local path wins even if it looks like a repo spec.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "EleutherAI" / "pythia-70m@step1").mkdir(parents=True)
    assert parse_spec("EleutherAI/pythia-70m@step1") == Path("EleutherAI/pythia-70m@step1")
    # Windows paths are never repo specs (they cannot exist inside the container).
    for windows in (r"C:\x", "C:/models/org@rev", r"\\server\share\model.safetensors"):
        with pytest.raises(InputError, match="Windows path"):
            parse_spec(windows)
    for unknown in ("pythia@step1", "a/b/c@d", "org/name@"):
        with pytest.raises(InputError, match="neither an existing local path nor a Hub source"):
            parse_spec(unknown)
    assert parse_spec("/data/missing.safetensors") == Path("/data/missing.safetensors")


@pytest.mark.integration
def test_download_resumes_verifies_and_keeps_the_token_off_the_cdn(
    fake_hub, rng, tmp_path, monkeypatch, caplog
):
    data = checkpoint_bytes(tmp_path, neox_tensors(rng))
    sha = fake_hub.add(REPO, "step1", data)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")
    caplog.set_level(logging.DEBUG)

    remote = fetch_metadata(HubSpec(REPO, "step1"))
    assert (remote.size, remote.sha256) == (len(data), sha)

    dest = tmp_path / "store" / "model.safetensors"
    fake_hub.drop_after[(REPO, "step1")] = 1000
    with pytest.raises(InputError, match="Run the command again to resume") as exc:
        download(remote, dest)
    assert exc.value.exit_code == 4
    part = dest.with_name("model.safetensors.part")
    assert part.stat().st_size == 1000 and not dest.exists()

    stats = download(remote, dest)
    assert stats.resumed_from == 1000
    assert dest.read_bytes() == data and not part.exists()

    cdn = [r for r in fake_hub.requests if r["server"] == "cdn"]
    hub_side = [r for r in fake_hub.requests if r["server"] == "hub"]
    assert cdn[-1]["headers"].get("Range") == "bytes=1000-"
    assert all(
        r["headers"].get("Authorization") == "Bearer hf_secret_token_value" for r in hub_side
    )
    assert all("Authorization" not in r["headers"] for r in cdn)  # never sent cross-host
    assert "hf_secret_token_value" not in caplog.text


@pytest.mark.integration
def test_sha256_mismatch_removes_the_partial_file(fake_hub, rng, tmp_path):
    fake_hub.add(REPO, "step1", checkpoint_bytes(tmp_path, neox_tensors(rng)))
    fake_hub.corrupt.add((REPO, "step1"))
    remote = fetch_metadata(HubSpec(REPO, "step1"))
    dest = tmp_path / "store" / "model.safetensors"
    with pytest.raises(InputError, match="failed verification") as exc:
        download(remote, dest)
    assert exc.value.exit_code == 4
    assert not dest.exists() and not dest.with_name("model.safetensors.part").exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_repo", "was not found on the Hub, or it is private or gated"),
        ("unknown_revision", "Revision 'step9' was not found"),
        ("bin_only", "only ships pytorch_model.bin"),
        ("network", "Network error contacting"),
        ("timeout", "Timed out after"),
    ],
)
def test_hub_errors_are_clear_input_errors(case, message, fake_hub, rng, tmp_path, monkeypatch):
    fake_hub.add(REPO, "step1", checkpoint_bytes(tmp_path, neox_tensors(rng)))
    fake_hub.bin_only.add((REPO, "old"))
    spec = {
        "unknown_repo": HubSpec("nobody/nothing", "main"),
        "unknown_revision": HubSpec(REPO, "step9"),
        "bin_only": HubSpec(REPO, "old"),
        "network": HubSpec(REPO, "step1"),
        "timeout": HubSpec(REPO, "step1"),
    }[case]
    if case == "network":
        monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")  # discard port: refused
    if case == "timeout":
        fake_hub.delay = 1.0
        monkeypatch.setattr(hub, "DEFAULT_TIMEOUT", 0.2)
    with pytest.raises(InputError, match=message) as exc:
        fetch_metadata(spec)
    assert exc.value.exit_code == 4
    if case == "bin_only":
        assert "D1" in str(exc.value)


@pytest.mark.integration
def test_rolling_store_limit_lru_pinning_and_stale_parts(fake_hub, rng, tmp_path):
    base = neox_tensors(rng)
    for i, rev in enumerate(("r1", "r2", "r3")):
        fake_hub.add(REPO, rev, checkpoint_bytes(tmp_path, perturb(base, 0.01 * (i + 1), rng)))
    remotes = {rev: fetch_metadata(HubSpec(REPO, rev)) for rev in ("r1", "r2", "r3")}

    max_seen = 0

    def counting_download(remote, dest):
        nonlocal max_seen
        stats = hub.download(remote, dest)
        max_seen = max(max_seen, len(store.complete()))
        return stats

    store = CheckpointStore(tmp_path / "ck", max_checkpoints=2, downloader=counting_download)
    stale = store.path_for(HubSpec(REPO, "r9")).with_name("model.safetensors.part")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"half a file")  # an interrupted run for another revision

    with store.use(remotes["r1"]), store.use(remotes["r2"]):
        assert len(store.complete()) == 2  # the .part file is never counted
        with pytest.raises(InputError, match="all of them are in use"), store.use(remotes["r3"]):
            pass
    with store.use(remotes["r3"]) as p3:  # evicts r1, the least recently used
        assert p3.read_bytes() == fake_hub.files[(REPO, "r3")]
    assert [p.parent.name for p in store.complete()] == ["r2", "r3"]
    assert not stale.exists()

    with store.use(remotes["r2"]), store.use(remotes["r1"]):  # r2 pinned: r3 is evicted
        pass
    assert [p.parent.name for p in store.complete()] == ["r1", "r2"]
    assert max_seen <= 2


@pytest.mark.integration
def test_store_cache_hit_skips_the_download(fake_hub, rng, tmp_path):
    fake_hub.add(REPO, "step1", checkpoint_bytes(tmp_path, neox_tensors(rng)))
    remote = fetch_metadata(HubSpec(REPO, "step1"))
    store = CheckpointStore(tmp_path / "ck")
    with store.use(remote):
        pass
    with store.use(remote) as path:
        assert path.is_file()
    assert fake_hub.cdn_gets() == ["step1"]  # downloaded exactly once

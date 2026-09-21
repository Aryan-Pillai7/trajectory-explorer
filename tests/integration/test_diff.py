import json
import math
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from conftest import neox_tensors, perturb
from trajectory_explorer import metrics, noise
from trajectory_explorer.diff import DiffOptions, DiffResult, diff_checkpoints


@pytest.fixture
def measure_calls(monkeypatch):
    """Count calls to the per-tensor measurement (the expensive step)."""
    calls = {"n": 0}
    real = metrics.measure_tensor

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "measure_tensor", counting)
    return calls


@pytest.mark.integration
def test_cache_hit_identical_shortcut_and_version_miss(
    make_checkpoint, rng, tmp_path, measure_calls, monkeypatch
):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", perturb(base, 0.01, rng))
    control_b = make_checkpoint("c.safetensors", perturb(base, 0.05, rng))
    opts = DiffOptions(cache_dir=tmp_path / "metrics")
    n = len(base)

    first = diff_checkpoints(a, b, opts)
    assert measure_calls["n"] == n
    assert first.n_tensors == n

    measure_calls["n"] = 0
    assert diff_checkpoints(a, b, opts) == first  # cache hit: nothing measured
    assert measure_calls["n"] == 0

    # Adding a control reuses the cached main pair; only the control pair is measured.
    with_control = diff_checkpoints(a, b, DiffOptions(opts.cache_dir, control=(a, control_b)))
    assert measure_calls["n"] == n
    assert all(m.status == noise.BELOW_CONTROL for m in with_control.tensors.values())

    # Byte-identical files short-circuit before any measurement.
    measure_calls["n"] = 0
    copy = tmp_path / "a_copy.safetensors"
    shutil.copyfile(a, copy)
    same = diff_checkpoints(a, copy, opts)
    assert same.identical and same.tensors == {} and measure_calls["n"] == 0
    assert {g.status for g in same.groups} == {noise.NO_CHANGE}

    # A new METRICS_VERSION must not reuse old cache entries.
    monkeypatch.setattr(metrics, "METRICS_VERSION", metrics.METRICS_VERSION + 1)
    diff_checkpoints(a, b, opts)
    assert measure_calls["n"] == n


@pytest.mark.integration
def test_local_hashes_are_memoized_per_path_size_and_mtime(make_checkpoint, rng, monkeypatch):
    from trajectory_explorer import diff

    calls = []
    real = diff.sha256_file
    monkeypatch.setattr(diff, "sha256_file", lambda p: calls.append(p) or real(p))
    path = make_checkpoint("a.safetensors", neox_tensors(rng))
    first = diff.local_source(path).info.sha256
    assert diff.local_source(path).info.sha256 == first and len(calls) == 1

    # Rewritten later (same size). A rewrite inside the same filesystem clock tick would keep
    # the mtime and reuse the old hash; within one CLI run files don't change, so we accept it.
    before = path.stat().st_mtime_ns
    make_checkpoint("a.safetensors", perturb(neox_tensors(rng), 0.01, rng))
    os.utime(path, ns=(before + 10**9, before + 10**9))
    assert diff.local_source(path).info.sha256 != first and len(calls) == 2


@pytest.mark.integration
def test_diff_result_json_round_trip_is_exact(make_checkpoint, rng, tmp_path):
    base = neox_tensors(rng)
    base["gpt_neox.layers.0.attention.dense.bias"] = np.zeros(8, np.float32)  # from zero
    moved = perturb(base, 0.01, rng)
    moved["gpt_neox.layers.0.attention.dense.bias"] = np.full(8, 0.01, np.float32)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", moved)
    c = make_checkpoint("c.safetensors", perturb(base, 0.001, rng))
    result = diff_checkpoints(a, b, DiffOptions(control=(a, c)))
    assert result.tensors["gpt_neox.layers.0.attention.dense.bias"].status == noise.FROM_ZERO

    text = result.to_json()
    assert json.loads(text)["schema_version"] == 2
    assert DiffResult.from_json(text) == result

    name = next(iter(result.tensors))
    bad_tensor = replace(result.tensors[name], abs_delta=float("nan"))
    with pytest.raises(ValueError):
        replace(result, tensors={**result.tensors, name: bad_tensor}).to_json()


@pytest.mark.integration
def test_group_relative_delta_uses_sums_of_squares(make_checkpoint):
    # One group (layer 0, attn_qkv) with two tensors of very different size.
    a = {
        "gpt_neox.layers.0.attention.query_key_value.weight": np.ones((24, 8), np.float32),
        "gpt_neox.layers.0.attention.query_key_value.bias": np.ones(24, np.float32),
    }
    b = {
        "gpt_neox.layers.0.attention.query_key_value.weight": np.full((24, 8), 1.1, np.float32),
        "gpt_neox.layers.0.attention.query_key_value.bias": np.full(24, 2.0, np.float32),
    }
    result = diff_checkpoints(make_checkpoint("a.safetensors", a), make_checkpoint("b.st", b))
    (group,) = result.groups
    # sqrt(192 * 0.1^2 + 24 * 1^2) / sqrt(192 + 24) = sqrt(0.12), not the mean ratio 0.55.
    assert group.rel_delta == pytest.approx(math.sqrt(0.12), rel=1e-6)
    assert (group.layer, group.component, group.status) == (0, "attn_qkv", noise.SIGNIFICANT)

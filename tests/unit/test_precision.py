import json
import warnings

import numpy as np
import pytest

from conftest import neox_tensors
from trajectory_explorer import metrics, noise
from trajectory_explorer.diff import DiffOptions, cache_path, diff_checkpoints
from trajectory_explorer.metrics import compute_tensor_metrics, sum_squares_and_f16

F16_FLOOR = noise.dtype_floor("F16", "F16")
F32_FLOOR = noise.dtype_floor("F32", "F32")


def f16_exact(values):
    """float32 values that are exactly float16 numbers (like Pythia's step files)."""
    return np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float32)


def one_f16_step(x):
    """Move every element one float16 step up: still exactly float16 values."""
    x16 = x.astype(np.float16)
    return np.nextafter(x16, np.float16(np.inf)).astype(np.float32)


@pytest.mark.unit
def test_float16_content_stored_as_float32_uses_the_float16_floor(rng):
    a = f16_exact(0.02 * rng.standard_normal((128, 64)))
    b = one_f16_step(a)
    m = compute_tensor_metrics(a, b, "F32", "F32")
    assert m.f16_rule and m.floor == pytest.approx(F16_FLOOR)
    assert F32_FLOOR < m.rel_delta < m.floor  # rounding-sized change: not significant
    assert m.status == noise.BELOW_FLOOR

    # The same-size change on ordinary float32 values is significant.
    a32 = (0.02 * rng.standard_normal((128, 64))).astype(np.float32)
    noise_ = rng.standard_normal(a32.shape).astype(np.float32)
    b32 = a32 + (m.rel_delta * np.linalg.norm(a32) / np.linalg.norm(noise_)) * noise_
    m32 = compute_tensor_metrics(a32, b32.astype(np.float32), "F32", "F32")
    assert not m32.f16_rule and m32.floor == pytest.approx(F32_FLOOR)
    assert m32.status == noise.SIGNIFICANT


@pytest.mark.unit
def test_only_one_float16_exact_side_keeps_the_float32_floor(rng):
    a = f16_exact(0.02 * rng.standard_normal((64, 32)))
    b = a + np.float32(1e-6) * rng.standard_normal(a.shape).astype(np.float32)
    m = compute_tensor_metrics(a, b.astype(np.float32), "F32", "F32")
    assert (m.exact_f16_a, m.exact_f16_b, m.f16_rule) == (True, False, False)
    assert m.floor == pytest.approx(F32_FLOOR)

    # A float16-stored side counts as float16 content; the F32 side must still be exact.
    m16 = compute_tensor_metrics(a.astype(np.float16).astype(np.float32), a, "F16", "F32")
    assert m16.f16_rule and m16.status == noise.NO_CHANGE


@pytest.mark.unit
def test_constant_tensor_behaves_sensibly():
    ones = np.ones(512, np.float32)
    same = compute_tensor_metrics(ones, ones.copy(), "F32", "F32")
    assert same.f16_rule and same.status == noise.NO_CHANGE
    step = compute_tensor_metrics(ones, one_f16_step(ones), "F32", "F32")  # 1 -> 1 + 2^-10
    assert step.rel_delta == pytest.approx(2.0**-10)
    assert step.status == noise.BELOW_FLOOR  # 9.8e-4 < 1.2e-3


@pytest.mark.unit
def test_nan_and_inf_never_crash_the_float16_check():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # overflow/invalid warnings would fail the test
        _, exact = sum_squares_and_f16(np.array([1.0, np.inf, -np.inf], np.float32), check_f16=True)
        assert exact is True
        _, exact = sum_squares_and_f16(np.array([1.0, np.nan], np.float32), check_f16=True)
        assert exact is True
        _, exact = sum_squares_and_f16(np.array([1e6, 0.5], np.float32), check_f16=True)
        assert exact is False  # beyond float16's range
    with pytest.raises(ValueError, match="NaN or inf"):  # still rejected as input, cleanly
        compute_tensor_metrics(
            np.array([1.0, np.nan], np.float32), np.ones(2, np.float32), "F32", "F32"
        )


@pytest.mark.unit
def test_an_old_metrics_version_cache_entry_misses(make_checkpoint, rng, tmp_path, monkeypatch):
    base = neox_tensors(rng)
    a = make_checkpoint("a.safetensors", base)
    b = make_checkpoint("b.safetensors", {n: t * np.float32(1.01) for n, t in base.items()})
    cache = tmp_path / "metrics"
    diff_checkpoints(a, b, DiffOptions(cache_dir=cache))
    (entry,) = cache.glob("*.json")
    # Turn it into a version-1 entry: old file name, old version, no precision fields.
    data = json.loads(entry.read_text())
    data["metrics_version"] = 1
    for m in data["measurements"].values():
        m.pop("exact_f16_a"), m.pop("exact_f16_b")
    old = entry.with_name(entry.name.replace(f"_m{metrics.METRICS_VERSION}.json", "_m1.json"))
    old.write_text(json.dumps(data))
    entry.unlink()

    calls = {"n": 0}
    real = metrics.measure_tensor

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "measure_tensor", counting)
    diff_checkpoints(a, b, DiffOptions(cache_dir=cache))
    assert calls["n"] == len(base)  # recomputed, not read from the v1 entry
    sha_a, sha_b = old.name.split("_")[:2]
    assert cache_path(cache, sha_a, sha_b).is_file()  # a fresh v2 entry exists

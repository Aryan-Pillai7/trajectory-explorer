import json

import numpy as np
import pytest

from trajectory_explorer import noise
from trajectory_explorer.metrics import (
    compute_tensor_metrics,
    entropy_effective_rank,
    gram_singular_values,
    random_erank_ratio,
)


def _metrics(a, b, dtype_a="F32", dtype_b="F32", **kwargs):
    return compute_tensor_metrics(a, b, dtype_a, dtype_b, **kwargs)


@pytest.mark.unit
def test_identical_inputs_give_zero_diff(rng):
    a = rng.standard_normal((64, 32)).astype(np.float32)
    m = _metrics(a, a.copy())
    assert m.abs_delta == 0.0
    assert m.rel_delta == 0.0
    assert m.status == noise.NO_CHANGE
    assert not m.significant
    assert (m.erank, m.r90, m.rank_label, m.concentration_label) == (None, None, "n/a", "n/a")


@pytest.mark.unit
def test_scaled_tensor_gives_known_relative_norm(rng):
    a = rng.standard_normal((50, 20)).astype(np.float32)
    m = _metrics(a, (a * np.float32(1.1)).astype(np.float32))
    assert m.rel_delta == pytest.approx(0.1, rel=1e-6)
    assert m.norm_b == pytest.approx(1.1 * m.norm_a, rel=1e-6)
    assert m.status == noise.SIGNIFICANT
    # B - A = 0.1 * A, so the delta has A's spectrum: a random, dense matrix.
    assert m.rank_label == "dense"


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(200, 64), (64, 200)], ids=["tall", "wide"])
def test_low_rank_delta_gives_expected_effective_rank(shape, rng):
    rows, cols = shape
    k = 5
    u, _ = np.linalg.qr(rng.standard_normal((rows, k)))
    v, _ = np.linalg.qr(rng.standard_normal((cols, k)))
    delta = (u @ v.T).astype(np.float32)  # k equal singular values of 1
    a = rng.standard_normal(shape).astype(np.float32)

    m = _metrics(a, a + delta)
    assert m.erank == pytest.approx(k, abs=1e-3)
    assert m.r90 == k
    assert m.rank_dims == min(shape)
    assert m.rank_label == "low-rank"

    dense = _metrics(a, a + 0.1 * rng.standard_normal(shape).astype(np.float32))
    assert dense.rank_label == "dense"


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(300, 40), (40, 300), (64, 64)])
def test_gram_spectrum_matches_full_svd(shape, rng):
    d = rng.standard_normal(shape).astype(np.float32)
    d[:, 0] *= 20.0  # uneven spectrum
    expected = np.linalg.svd(d.astype(np.float64), compute_uv=False)
    # A tiny chunk size forces many accumulation steps.
    got = gram_singular_values(d, chunk_elements=97)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6 * expected[0])
    assert entropy_effective_rank(got) == pytest.approx(entropy_effective_rank(expected), rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(256, 1024), (256, 256)])
def test_random_baseline_matches_monte_carlo(shape, rng):
    sv = np.linalg.svd(rng.standard_normal(shape), compute_uv=False)
    observed = entropy_effective_rank(sv) / min(shape)
    assert random_erank_ratio(*shape) == pytest.approx(observed, rel=0.02)


@pytest.mark.unit
def test_float16_rounding_is_below_floor_and_one_percent_is_above(rng):
    a = (0.02 * rng.standard_normal((256, 128))).astype(np.float32)

    rounded = _metrics(a, a.astype(np.float16).astype(np.float32), "F32", "F16")
    assert 0.0 < rounded.rel_delta < rounded.floor
    assert rounded.status == noise.BELOW_FLOOR

    noise_1pct = rng.standard_normal(a.shape).astype(np.float32)
    noise_1pct *= 0.01 * np.linalg.norm(a) / np.linalg.norm(noise_1pct)
    moved = _metrics(a, a + noise_1pct)
    assert moved.rel_delta == pytest.approx(0.01, rel=1e-3)
    assert moved.status == noise.SIGNIFICANT
    # A control (reference scale) larger than the delta mutes it, separately from the floor.
    assert _metrics(a, a + noise_1pct, control=0.05).status == noise.BELOW_CONTROL


@pytest.mark.unit
def test_concentration_labels(rng):
    a = rng.standard_normal((100, 32)).astype(np.float32)
    one_row = a.copy()
    one_row[17] += 1.0
    assert _metrics(a, one_row).concentration_label == "concentrated"
    assert _metrics(a, one_row).top_row_share == pytest.approx(1.0)

    spread = a + 0.1 * rng.standard_normal(a.shape).astype(np.float32)
    assert _metrics(a, spread).concentration_label == "spread"


@pytest.mark.unit
def test_zero_reference_is_reported_without_nan_or_inf(rng):
    zeros = np.zeros(64, dtype=np.float32)  # e.g. a bias at step 0
    b = rng.standard_normal(64).astype(np.float32)

    m = _metrics(zeros, b)
    assert m.rel_delta is None
    assert m.from_zero
    assert m.status == noise.FROM_ZERO
    assert m.significant
    json.dumps(m.to_dict(), allow_nan=False)  # raises on NaN or inf

    both_zero = _metrics(zeros, zeros.copy())
    assert both_zero.status == noise.NO_CHANGE
    json.dumps(both_zero.to_dict(), allow_nan=False)

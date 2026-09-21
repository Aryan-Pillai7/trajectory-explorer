from pathlib import Path

import numpy as np
import pytest

from conftest import from_bf16_bits, to_bf16_bits
from trajectory_explorer.errors import CheckpointFormatError, InputError, UnsupportedDtypeError
from trajectory_explorer.reader import Checkpoint


@pytest.mark.unit
def test_decodes_f32_f16_bf16_to_float32(make_checkpoint, rng):
    w32 = rng.standard_normal((3, 5)).astype(np.float32)
    w16 = rng.standard_normal((4,)).astype(np.float16)
    bf16_bits = to_bf16_bits(rng.standard_normal((2, 3)))
    path = make_checkpoint(
        "mixed.safetensors", {"a.f32": w32, "b.f16": w16, "c.bf16": ("BF16", bf16_bits)}
    )

    with Checkpoint(path) as ckpt:
        # Header-only inspection: dtypes and shapes as stored.
        assert ckpt.names() == ["a.f32", "b.f16", "c.bf16"]
        assert [ckpt.spec(n).dtype for n in ckpt.names()] == ["F32", "F16", "BF16"]
        assert ckpt.spec("c.bf16").shape == (2, 3)

        loaded = {n: ckpt.load(n) for n in ckpt.names()}

    assert all(t.dtype == np.float32 for t in loaded.values())
    np.testing.assert_array_equal(loaded["a.f32"], w32)
    np.testing.assert_array_equal(loaded["b.f16"], w16.astype(np.float32))
    # BF16 -> float32 widening is exact.
    np.testing.assert_array_equal(loaded["c.bf16"], from_bf16_bits(bf16_bits).reshape(2, 3))


@pytest.mark.unit
@pytest.mark.parametrize("case", ["missing", "garbage", "non_float"])
def test_bad_inputs_raise_clear_errors(case, tmp_path: Path, make_checkpoint):
    if case == "missing":
        with pytest.raises(InputError, match="not found"):
            Checkpoint(tmp_path / "nope.safetensors")
    elif case == "garbage":
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\xff" * 64)
        with pytest.raises(CheckpointFormatError):
            Checkpoint(bad)
    else:
        path = make_checkpoint("mask.safetensors", {"mask": np.ones((2, 2), dtype=np.uint8)})
        with Checkpoint(path) as ckpt, pytest.raises(UnsupportedDtypeError, match="U8"):
            ckpt.load("mask")

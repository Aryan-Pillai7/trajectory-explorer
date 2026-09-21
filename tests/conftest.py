"""Shared fixtures: tiny synthetic safetensors checkpoints written by hand.

Writing the format directly (instead of via safetensors.numpy.save_file) lets tests create
BF16 tensors, which numpy has no dtype for.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

_NUMPY_TO_ST = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("uint8"): "U8",
}


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Truncate float32 values to bfloat16 bit patterns (uint16)."""
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


TensorInput = np.ndarray | tuple[str, np.ndarray]


def write_safetensors(path: Path, tensors: Mapping[str, TensorInput]) -> Path:
    """Write a safetensors file. Values are arrays, or ("BF16", uint16_bits) tuples."""
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        if isinstance(value, tuple):
            dtype, array = value
            data = np.ascontiguousarray(array, dtype="<u2").tobytes()
        else:
            array = np.ascontiguousarray(value)
            dtype = _NUMPY_TO_ST[array.dtype]
            data = array.astype(array.dtype.newbyteorder("<")).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    return path


def neox_tensors(
    rng: np.random.Generator, *, layers: int = 2, hidden: int = 8, vocab: int = 40
) -> dict[str, np.ndarray]:
    """A tiny GPT-NeoX (Pythia-style) parameter set, float32, same names as the real model."""

    def w(*shape: int) -> np.ndarray:
        return (0.02 * rng.standard_normal(shape)).astype(np.float32)

    t = {"gpt_neox.embed_in.weight": w(vocab, hidden)}
    for i in range(layers):
        p = f"gpt_neox.layers.{i}."
        t |= {
            p + "input_layernorm.weight": (1 + w(hidden)).astype(np.float32),
            p + "input_layernorm.bias": w(hidden),
            p + "post_attention_layernorm.weight": (1 + w(hidden)).astype(np.float32),
            p + "post_attention_layernorm.bias": w(hidden),
            p + "attention.query_key_value.weight": w(3 * hidden, hidden),
            p + "attention.query_key_value.bias": w(3 * hidden),
            p + "attention.dense.weight": w(hidden, hidden),
            p + "attention.dense.bias": w(hidden),
            p + "mlp.dense_h_to_4h.weight": w(4 * hidden, hidden),
            p + "mlp.dense_h_to_4h.bias": w(4 * hidden),
            p + "mlp.dense_4h_to_h.weight": w(hidden, 4 * hidden),
            p + "mlp.dense_4h_to_h.bias": w(hidden),
        }
    t |= {
        "gpt_neox.final_layer_norm.weight": (1 + w(hidden)).astype(np.float32),
        "gpt_neox.final_layer_norm.bias": w(hidden),
        "embed_out.weight": w(vocab, hidden),
    }
    return t


def perturb(
    tensors: dict[str, np.ndarray], rel: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Add Gaussian noise with relative Frobenius norm ``rel`` to every tensor."""
    out = {}
    for name, value in tensors.items():
        noise = rng.standard_normal(value.shape).astype(np.float32)
        noise *= rel * np.linalg.norm(value) / np.linalg.norm(noise)
        out[name] = (value + noise).astype(np.float32)
    return out


@pytest.fixture
def make_checkpoint(tmp_path: Path) -> Callable[[str, Mapping[str, TensorInput]], Path]:
    def _make(filename: str, tensors: Mapping[str, TensorInput]) -> Path:
        return write_safetensors(tmp_path / filename, tensors)

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)

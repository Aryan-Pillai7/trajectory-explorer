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


@pytest.fixture
def make_checkpoint(tmp_path: Path) -> Callable[[str, Mapping[str, TensorInput]], Path]:
    def _make(filename: str, tensors: Mapping[str, TensorInput]) -> Path:
        return write_safetensors(tmp_path / filename, tensors)

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)

"""Lazy, one-tensor-at-a-time reader for safetensors checkpoints.

The header (names, dtypes, shapes, byte offsets) is parsed directly, so inspecting a
checkpoint never touches weight bytes. ``Checkpoint.load`` reads a single tensor and returns
it as float32. F32/F16/F64 go through the safetensors library; BF16 is decoded from raw bytes
because safetensors' numpy backend rejects it (measured: safetensors 0.8.0 + numpy 2.5.3 raise
``TypeError: data type 'bfloat16' not understood``). BF16 is the top half of a float32, so
widening the 16 bits into the high half of a uint32 is exact.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import numpy as np
from safetensors import safe_open

from trajectory_explorer.errors import CheckpointFormatError, InputError, UnsupportedDtypeError

# Bytes per element for every dtype the safetensors format defines that we may meet.
DTYPE_SIZES: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}  # fmt: skip
FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
_MAX_HEADER_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class TensorSpec:
    """Header information for one tensor. No weight data."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]  # byte range relative to the start of the data section

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def is_float(self) -> bool:
        return self.dtype in FLOAT_DTYPES


def read_header(path: Path) -> tuple[dict[str, TensorSpec], dict[str, str], int]:
    """Parse a safetensors header. Returns (specs, metadata, data_start_offset)."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise CheckpointFormatError(f"{path}: file too short to be safetensors")
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len > min(_MAX_HEADER_BYTES, file_size - 8):
                raise CheckpointFormatError(f"{path}: invalid safetensors header length")
            header = json.loads(fh.read(header_len))
    except FileNotFoundError:
        raise InputError(f"Checkpoint not found: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError(f"{path}: cannot read safetensors header ({exc})") from exc

    if not isinstance(header, dict):
        raise CheckpointFormatError(f"{path}: safetensors header is not a JSON object")
    metadata = header.pop("__metadata__", None) or {}
    data_start = 8 + header_len
    data_size = file_size - data_start
    specs: dict[str, TensorSpec] = {}
    for name, entry in header.items():
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(s) for s in entry["shape"])
            begin, end = (int(o) for o in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError(f"{path}: malformed header entry for {name!r}") from exc
        itemsize = DTYPE_SIZES.get(dtype)
        if itemsize is not None and end - begin != math.prod(shape) * itemsize:
            raise CheckpointFormatError(f"{path}: byte size of {name!r} does not match its shape")
        if not 0 <= begin <= end <= data_size:
            raise CheckpointFormatError(f"{path}: data offsets of {name!r} are out of range")
        specs[name] = TensorSpec(name, dtype, shape, (begin, end))
    return specs, metadata, data_start


class Checkpoint:
    """A safetensors file opened for lazy, per-tensor float32 reads.

    Use as a context manager. Only one tensor is materialised per ``load`` call.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._specs, self.metadata, self._data_start = read_header(self.path)
        self._st = None  # safetensors handle, opened lazily on first non-BF16 load
        self._raw: BinaryIO | None = None

    # -- header-only inspection -------------------------------------------------------
    def specs(self) -> dict[str, TensorSpec]:
        return dict(self._specs)

    def names(self) -> list[str]:
        return list(self._specs)

    def spec(self, name: str) -> TensorSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None

    # -- data -------------------------------------------------------------------------
    def load(self, name: str) -> np.ndarray:
        """Read one tensor and return it as a C-contiguous float32 array."""
        spec = self.spec(name)
        if not spec.is_float:
            raise UnsupportedDtypeError(
                f"{self.path}: tensor {name!r} has non-float dtype {spec.dtype}"
            )
        if spec.dtype == "BF16":
            return self._load_bf16(spec)
        if self._st is None:
            self._st = safe_open(str(self.path), framework="np")
        array = self._st.get_tensor(name)
        return np.ascontiguousarray(array, dtype=np.float32)

    def _load_bf16(self, spec: TensorSpec) -> np.ndarray:
        if self._raw is None:
            self._raw = self.path.open("rb")
        self._raw.seek(self._data_start + spec.offsets[0])
        bits = np.fromfile(self._raw, dtype="<u2", count=spec.numel)
        if bits.size != spec.numel:
            raise CheckpointFormatError(f"{self.path}: truncated data for {spec.name!r}")
        return (bits.astype(np.uint32) << 16).view(np.float32).reshape(spec.shape)

    # -- lifecycle --------------------------------------------------------------------
    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        self._st = None

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Checkpoint({str(self.path)!r}, tensors={len(self._specs)})"

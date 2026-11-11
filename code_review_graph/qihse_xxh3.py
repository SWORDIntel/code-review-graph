"""Lazy native XXH3-128 binding for versioned QIHSE graph identifiers.

Backend selection happens once. Missing native support fails export rather than
silently changing the algorithm. Public output is upstream canonical hex.
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable


class _Binding:
    def __init__(self) -> None:
        configured = os.environ.get("QIHSE_CODEGRAPH_XXH3_LIBRARY")
        path = Path(configured) if configured else (
            Path.home() / ".local/lib/qihse-codegraph/current/libqihse_xxh3.so"
        )
        try:
            self.library = ctypes.CDLL(str(path), use_errno=True)
            self.library.qx3_backend_get.argtypes = [ctypes.c_char_p]
            self.library.qx3_backend_get.restype = ctypes.c_void_p
            self.library.qx3_hash_canonical128.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ]
            self.library.qx3_hash_canonical128.restype = ctypes.c_int
            self.library.qx3_batch_canonical128.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_void_p,
            ]
            self.library.qx3_batch_canonical128.restype = ctypes.c_int
        except (OSError, AttributeError) as exc:
            raise RuntimeError(
                "QIHSE graph export requires the tuned XXH3 native library; "
                "set QIHSE_CODEGRAPH_XXH3_LIBRARY or install the CodeGraph runtime bundle"
            ) from exc
        backend = os.environ.get("QIHSE_CODEGRAPH_XXH3_BACKEND", "auto")
        self.backend = self.library.qx3_backend_get(backend.encode("ascii"))
        if not self.backend:
            raise RuntimeError(f"QIHSE XXH3 backend is unavailable: {backend}")
        if self.one(b"abc") != "06b05ab6733a618578af5f94892f3950":
            raise RuntimeError("QIHSE XXH3 native library failed its canonical known-answer check")

    def one(self, data: bytes) -> str:
        output = (ctypes.c_ubyte * 16)()
        if self.library.qx3_hash_canonical128(self.backend, data, len(data), output):
            raise RuntimeError("QIHSE XXH3 hashing failed")
        return bytes(output).hex()

    def batch(self, rows: list[bytes]) -> list[str]:
        if not rows:
            return []
        offsets = (ctypes.c_size_t * (len(rows) + 1))()
        size = 0
        for i, row in enumerate(rows, 1):
            size += len(row)
            offsets[i] = size
        blob = b"".join(rows)
        output = (ctypes.c_ubyte * (len(rows) * 16))()
        if self.library.qx3_batch_canonical128(
            self.backend, blob, len(blob), offsets, len(rows), output,
        ):
            raise RuntimeError("QIHSE XXH3 batch hashing failed")
        encoded = bytes(output).hex()
        return [encoded[i:i + 32] for i in range(0, len(encoded), 32)]


@lru_cache(maxsize=1)
def _binding() -> _Binding:
    return _Binding()


def hash_one(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("XXH3 graph identity must be bytes")
    return _binding().one(data)


def hash_many(data: Iterable[bytes]) -> list[str]:
    """Bound native bridge batches to 4096 records or 8 MiB of input."""
    result: list[str] = []
    chunk: list[bytes] = []
    size = 0
    binding = _binding()
    for row in data:
        if not isinstance(row, bytes):
            raise TypeError("XXH3 graph identity must be bytes")
        if chunk and (len(chunk) == 4096 or size + len(row) > 8 * 1024 * 1024):
            result.extend(binding.batch(chunk))
            chunk, size = [], 0
        chunk.append(row)
        size += len(row)
    result.extend(binding.batch(chunk))
    return result

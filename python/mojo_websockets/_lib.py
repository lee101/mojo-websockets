"""Build and load the Mojo WebSocket kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "websockets.mojo"
LIBRARY = Path(
    os.environ.get(
        "MOJO_WEBSOCKETS_LIB",
        ROOT / "dist" / "libmojo-websockets.so",
    )
)

I = ctypes.c_int64
P = ctypes.c_void_p
_SIGNATURES = {
    "mws_apply_mask": ([P, P, I, I, I, I, I], I),
    "mws_serialize_frame": ([P, P, I, I, I, I, I, I, I], I),
}

_allocate_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_allocate_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_allocate_bytes.restype = ctypes.py_object
_bytes_address = ctypes.pythonapi.PyBytes_AsString
_bytes_address.argtypes = [ctypes.py_object]
_bytes_address.restype = ctypes.c_void_p

_library: ctypes.CDLL | None = None
_cpu_device: int | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> Path:
    explicit = "MOJO_WEBSOCKETS_LIB" in os.environ
    if LIBRARY.exists() and (
        explicit or (not force and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime)
    ):
        return LIBRARY
    if explicit:
        raise BuildError(f"MOJO_WEBSOCKETS_LIB doesn't exist: {LIBRARY}")
    proc = subprocess.run(
        ["bash", str(ROOT / "build" / "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not LIBRARY.exists():
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def ensure_parallel_runtime() -> bool:
    global _cpu_device
    if _cpu_device is not None:
        return bool(_cpu_device)
    try:
        initialize = lib().KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        initialize.argtypes = []
        initialize.restype = ctypes.c_void_p
        _cpu_device = int(initialize() or 0)
    except (AttributeError, OSError):
        _cpu_device = 0
    return bool(_cpu_device)


def source_buffer(data: object) -> tuple[object, int, int]:
    """Return an owner, byte address, and byte length for a contiguous buffer."""
    if isinstance(data, bytes):
        return data, int(_bytes_address(data) or 0), len(data)
    view = memoryview(data)
    if not view.c_contiguous or view.readonly:
        copied = bytes(view)
        return copied, int(_bytes_address(copied) or 0), len(copied)
    byte_view = view.cast("B")
    array = (ctypes.c_char * byte_view.nbytes).from_buffer(byte_view)
    # array retains byte_view, which retains the original exporter for the call.
    return array, ctypes.addressof(array), byte_view.nbytes


def new_bytes(size: int) -> bytes:
    if size < 0:
        raise ValueError("byte allocation size must be non-negative")
    return _allocate_bytes(None, size)


def bytes_address(data: bytes) -> int:
    return int(_bytes_address(data) or 0)

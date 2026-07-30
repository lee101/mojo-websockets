"""WebSocket byte utilities."""

from __future__ import annotations

import sys

from ._lib import bytes_address, ensure_parallel_runtime, lib, new_bytes, source_buffer

BytesLike = bytes | bytearray | memoryview
_PARALLEL_THRESHOLD = 4 * 1024 * 1024


def apply_mask(data: BytesLike, mask: bytes | bytearray) -> bytes:
    """Apply a four-byte repeating WebSocket mask."""
    if len(mask) != 4:
        raise ValueError("mask must contain 4 bytes")
    length = len(data) if isinstance(data, bytes) else memoryview(data).nbytes
    if not length:
        return b""
    if length <= 512:
        source_bytes = bytes(data)
        repeated = mask * (length // 4) + mask[: length % 4]
        value = int.from_bytes(source_bytes, sys.byteorder) ^ int.from_bytes(
            repeated, sys.byteorder
        )
        return value.to_bytes(length, sys.byteorder)
    source_owner, source, length = source_buffer(data)
    destination = new_bytes(length)
    destination_address = bytes_address(destination)
    packed_mask = int.from_bytes(mask, "little")
    use_parallel = int(
        length >= _PARALLEL_THRESHOLD and ensure_parallel_runtime()
    )
    status = lib().mws_apply_mask(
        source,
        destination_address,
        length,
        length,
        len(destination),
        packed_mask,
        use_parallel,
    )
    # Keep the buffer exporter and its ctypes view alive through the native call.
    _ = source_owner
    if status != 0:
        raise RuntimeError(f"Mojo mask kernel failed with status {status}")
    return destination

"""Benchmark Mojo masking and framing against websockets."""

from __future__ import annotations

import gc
import platform
import statistics
import time
from pathlib import Path

import mojo_websockets.frames as mojo_frames
import websockets.frames as upstream_frames


def machine() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def measure(function, minimum_time: float = 0.2) -> float:
    loops = 1
    while True:
        started = time.perf_counter()
        for _ in range(loops):
            function()
        elapsed = time.perf_counter() - started
        if elapsed >= minimum_time:
            break
        loops *= 2
    samples = []
    gc.disable()
    try:
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(loops):
                function()
            samples.append((time.perf_counter() - started) / loops)
    finally:
        gc.enable()
    return statistics.median(samples)


def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e3:.2f} ms"


def main() -> None:
    print(f"Machine: {machine()} ({platform.system()} {platform.machine()})")
    print()
    print("| operation | Mojo | upstream websockets | speedup |")
    print("| --- | ---: | ---: | ---: |")

    key = b"\x12\x34\x56\x78"
    for size in (64, 4096, 1 << 20, 16 << 20):
        payload = bytes((index * 17 + 3) & 255 for index in range(size))
        mojo_time = measure(lambda: mojo_frames.apply_mask(payload, key))
        upstream_time = measure(lambda: upstream_frames.apply_mask(payload, key))
        print(
            f"| `apply_mask` {size:,} B | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {upstream_time / mojo_time:.2f}x |"
        )

    payload = bytes((index * 29 + 5) & 255 for index in range(1 << 20))
    fixed_key = lambda size: key
    original_token_bytes = mojo_frames.secrets.token_bytes
    mojo_frames.secrets.token_bytes = fixed_key
    try:
        mojo_unmasked = mojo_frames.Frame(mojo_frames.OP_BINARY, payload)
        upstream_unmasked = upstream_frames.Frame(upstream_frames.OP_BINARY, payload)
        mojo_time = measure(lambda: mojo_unmasked.serialize(mask=False))
        upstream_time = measure(lambda: upstream_unmasked.serialize(mask=False))
        print(
            f"| serialize unmasked 1 MiB | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {upstream_time / mojo_time:.2f}x |"
        )

        mojo_time = measure(lambda: mojo_unmasked.serialize(mask=True))
        upstream_time = measure(lambda: upstream_unmasked.serialize(mask=True))
        print(
            f"| serialize masked 1 MiB | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {upstream_time / mojo_time:.2f}x |"
        )
    finally:
        mojo_frames.secrets.token_bytes = original_token_bytes


if __name__ == "__main__":
    main()

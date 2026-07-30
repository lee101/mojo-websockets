"""Base interface for WebSocket extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..frames import Frame


class Extension:
    name: str

    def decode(self, frame: Frame, *, max_size: int | None = None) -> Frame:
        raise NotImplementedError

    def encode(self, frame: Frame) -> Frame:
        raise NotImplementedError

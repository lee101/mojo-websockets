"""Exceptions raised by the Sans-I/O framing API."""

from __future__ import annotations


class WebSocketException(Exception):
    pass


class ProtocolError(WebSocketException):
    pass


class PayloadTooBig(WebSocketException):
    def __init__(
        self,
        size_or_message: int | None | str,
        max_size: int | None = None,
        current_size: int | None = None,
    ) -> None:
        if isinstance(size_or_message, str):
            self.message: str | None = size_or_message
        else:
            self.message = None
            self.size = size_or_message
            if max_size is None:
                raise AssertionError("max_size is required")
            self.max_size = max_size
            self.current_size: int | None = None
            self.set_current_size(current_size)

    def __str__(self) -> str:
        if self.message is not None:
            return self.message
        message = "frame "
        if self.size is not None:
            message += f"with {self.size} bytes "
        if self.current_size is not None:
            message += f"after reading {self.current_size} bytes "
        return message + f"exceeds limit of {self.max_size} bytes"

    def set_current_size(self, current_size: int | None) -> None:
        if self.current_size is not None:
            raise AssertionError("current size is already set")
        if current_size is not None:
            self.max_size += current_size
            self.current_size = current_size

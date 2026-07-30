"""RFC 6455 frame types and Sans-I/O parsing."""

from __future__ import annotations

import dataclasses
import enum
import os
import secrets
import struct
from collections.abc import Generator, Sequence
from typing import Callable

from ._lib import bytes_address, ensure_parallel_runtime, lib, new_bytes
from .exceptions import PayloadTooBig, ProtocolError
from .utils import BytesLike, apply_mask

__all__ = [
    "Opcode",
    "OP_CONT",
    "OP_TEXT",
    "OP_BINARY",
    "OP_CLOSE",
    "OP_PING",
    "OP_PONG",
    "DATA_OPCODES",
    "CTRL_OPCODES",
    "CloseCode",
    "Frame",
    "Close",
]


class Opcode(enum.IntEnum):
    CONT, TEXT, BINARY = 0x00, 0x01, 0x02
    CLOSE, PING, PONG = 0x08, 0x09, 0x0A


OP_CONT = Opcode.CONT
OP_TEXT = Opcode.TEXT
OP_BINARY = Opcode.BINARY
OP_CLOSE = Opcode.CLOSE
OP_PING = Opcode.PING
OP_PONG = Opcode.PONG
DATA_OPCODES = OP_CONT, OP_TEXT, OP_BINARY
CTRL_OPCODES = OP_CLOSE, OP_PING, OP_PONG


class CloseCode(enum.IntEnum):
    NORMAL_CLOSURE = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    NO_STATUS_RCVD = 1005
    ABNORMAL_CLOSURE = 1006
    INVALID_DATA = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    MANDATORY_EXTENSION = 1010
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012
    TRY_AGAIN_LATER = 1013
    BAD_GATEWAY = 1014
    TLS_HANDSHAKE = 1015


CLOSE_CODE_EXPLANATIONS: dict[int, str] = {
    CloseCode.NORMAL_CLOSURE: "OK",
    CloseCode.GOING_AWAY: "going away",
    CloseCode.PROTOCOL_ERROR: "protocol error",
    CloseCode.UNSUPPORTED_DATA: "unsupported data",
    CloseCode.NO_STATUS_RCVD: "no status received [internal]",
    CloseCode.ABNORMAL_CLOSURE: "abnormal closure [internal]",
    CloseCode.INVALID_DATA: "invalid frame payload data",
    CloseCode.POLICY_VIOLATION: "policy violation",
    CloseCode.MESSAGE_TOO_BIG: "message too big",
    CloseCode.MANDATORY_EXTENSION: "mandatory extension",
    CloseCode.INTERNAL_ERROR: "internal error",
    CloseCode.SERVICE_RESTART: "service restart",
    CloseCode.TRY_AGAIN_LATER: "try again later",
    CloseCode.BAD_GATEWAY: "bad gateway",
    CloseCode.TLS_HANDSHAKE: "TLS handshake failure [internal]",
}

EXTERNAL_CLOSE_CODES = {
    CloseCode.NORMAL_CLOSURE,
    CloseCode.GOING_AWAY,
    CloseCode.PROTOCOL_ERROR,
    CloseCode.UNSUPPORTED_DATA,
    CloseCode.INVALID_DATA,
    CloseCode.POLICY_VIOLATION,
    CloseCode.MESSAGE_TOO_BIG,
    CloseCode.MANDATORY_EXTENSION,
    CloseCode.INTERNAL_ERROR,
    CloseCode.SERVICE_RESTART,
    CloseCode.TRY_AGAIN_LATER,
    CloseCode.BAD_GATEWAY,
}

OK_CLOSE_CODES = {
    CloseCode.NORMAL_CLOSURE,
    CloseCode.GOING_AWAY,
    CloseCode.NO_STATUS_RCVD,
}


def is_utf8_fragment(
    data: bytes,
    must_start_clean: bool = False,
    must_end_clean: bool = False,
) -> bool:
    start, end = 0, len(data)
    if not must_start_clean:
        max_start = min(3, len(data))
        while start < max_start and data[start] & 0xC0 == 0x80:
            start += 1
    if not must_end_clean and data:
        end -= 1
        min_end = max(len(data) - 4, start)
        while end >= min_end:
            byte = data[end]
            if byte & 0xC0 == 0x80:
                end -= 1
                continue
            if byte & 0x80 == 0:
                sequence_length = 1
            elif byte & 0xE0 == 0xC0:
                sequence_length = 2
            elif byte & 0xF0 == 0xE0:
                sequence_length = 3
            elif byte & 0xF8 == 0xF0:
                sequence_length = 4
            else:
                sequence_length = 0
            if sequence_length <= len(data) - end:
                end = len(data)
            break
    try:
        text = data[start:end].decode()
    except UnicodeDecodeError:
        return False
    return "\\x" not in repr(text)


def _serialize(payload: bytes, head1: int, mask: bool, mask_bytes: bytes) -> bytes:
    length = len(payload)
    if not mask:
        if length < 126:
            header = struct.pack("!BB", head1, length)
        elif length < 65536:
            header = struct.pack("!BBH", head1, 126, length)
        else:
            header = struct.pack("!BBQ", head1, 127, length)
        return header + payload
    header_size = 2 if length < 126 else 4 if length < 65536 else 10
    destination = new_bytes(header_size + 4 + length)
    packed_mask = int.from_bytes(mask_bytes, "little")
    use_parallel = int(length >= 4 * 1024 * 1024 and ensure_parallel_runtime())
    written = lib().mws_serialize_frame(
        bytes_address(payload) if payload else 0,
        bytes_address(destination),
        length,
        length,
        len(destination),
        head1,
        1,
        packed_mask,
        use_parallel,
    )
    if written != len(destination):
        raise RuntimeError("Mojo frame serializer returned an invalid size")
    return destination


@dataclasses.dataclass
class Frame:
    opcode: Opcode
    data: BytesLike
    fin: bool = True
    rsv1: bool = False
    rsv2: bool = False
    rsv3: bool = False

    MAX_LOG_SIZE = int(os.environ.get("WEBSOCKETS_MAX_LOG_SIZE", "75"))
    DEFAULT_IS_TEXT = {OP_TEXT: True, OP_BINARY: False, OP_CLOSE: True}

    @classmethod
    def parse(
        cls,
        read_exact: Callable[[int], Generator[None, None, bytes | bytearray]],
        *,
        mask: bool,
        max_size: int | None = None,
        extensions: Sequence[extensions.Extension] | None = None,
    ) -> Generator[None, None, Frame]:
        data = yield from read_exact(2)
        head1, head2 = struct.unpack("!BB", data)
        fin = bool(head1 & 0x80)
        rsv1 = bool(head1 & 0x40)
        rsv2 = bool(head1 & 0x20)
        rsv3 = bool(head1 & 0x10)
        try:
            opcode = Opcode(head1 & 0x0F)
        except ValueError as exc:
            raise ProtocolError("invalid opcode") from exc
        if bool(head2 & 0x80) != mask:
            raise ProtocolError("incorrect masking")
        length = head2 & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", (yield from read_exact(2)))
        elif length == 127:
            (length,) = struct.unpack("!Q", (yield from read_exact(8)))
        if max_size is not None and length > max_size:
            raise PayloadTooBig(length, max_size)
        if mask:
            mask_bytes = yield from read_exact(4)
        data = yield from read_exact(length)
        if mask:
            data = apply_mask(data, mask_bytes)
        frame = cls(opcode, data, fin, rsv1, rsv2, rsv3)
        for extension in reversed(extensions or []):
            frame = extension.decode(frame, max_size=max_size)
        frame.check()
        return frame

    def serialize(
        self,
        *,
        mask: bool,
        extensions: Sequence[extensions.Extension] | None = None,
    ) -> bytes:
        self.check()
        frame = self
        for extension in extensions or []:
            frame = extension.encode(frame)
        head1 = (
            (0x80 if frame.fin else 0)
            | (0x40 if frame.rsv1 else 0)
            | (0x20 if frame.rsv2 else 0)
            | (0x10 if frame.rsv3 else 0)
            | int(frame.opcode)
        )
        payload = bytes(frame.data)
        mask_bytes = secrets.token_bytes(4) if mask else b""
        return _serialize(payload, head1, mask, mask_bytes)

    def check(self) -> None:
        if self.rsv1 or self.rsv2 or self.rsv3:
            raise ProtocolError("reserved bits must be 0")
        if self.opcode in CTRL_OPCODES:
            if len(self.data) > 125:
                raise ProtocolError("control frame too long")
            if not self.fin:
                raise ProtocolError("fragmented control frame")

    def __str__(self) -> str:
        expected_text = self.DEFAULT_IS_TEXT.get(self.opcode)
        data_repr, is_text = self._data_repr()
        data_type = "" if expected_text == is_text else ("text" if is_text else "binary")
        length = f"{len(self.data)} byte{'' if len(self.data) == 1 else 's'}"
        non_final = "" if self.fin else "continued"
        metadata = ", ".join(filter(None, [data_type, length, non_final]))
        return f"{self.opcode.name} {data_repr} [{metadata}]"

    def _data_repr(self) -> tuple[str, bool | None]:
        if not self.data:
            return "''", self.DEFAULT_IS_TEXT.get(self.opcode)
        if self.opcode is OP_CLOSE:
            try:
                return str(Close.parse(self.data)), True
            except (ProtocolError, UnicodeDecodeError):
                pass
        raw = bytes(self.data)
        is_text = is_utf8_fragment(
            raw,
            must_start_clean=self.opcode != OP_CONT,
            must_end_clean=self.fin,
        )
        if is_text:
            data_repr = repr(raw.decode(errors="replace"))
        else:
            binary = raw
            if len(binary) > self.MAX_LOG_SIZE // 3:
                cut = (self.MAX_LOG_SIZE // 3 - 1) // 3
                binary = binary[: 2 * cut] + b"\x00\x00" + binary[-cut:]
            data_repr = binary.hex(" ")
        if len(data_repr) > self.MAX_LOG_SIZE:
            cut = self.MAX_LOG_SIZE // 3 - 1
            data_repr = data_repr[: 2 * cut] + "..." + data_repr[-cut:]
        return data_repr, is_text


@dataclasses.dataclass
class Close:
    code: CloseCode | int
    reason: str

    def __str__(self) -> str:
        if 3000 <= self.code < 4000:
            explanation = "registered"
        elif 4000 <= self.code < 5000:
            explanation = "private use"
        else:
            explanation = CLOSE_CODE_EXPLANATIONS.get(self.code, "unknown")
        result = f"{self.code} ({explanation})"
        return f"{result} {self.reason}" if self.reason else result

    @classmethod
    def parse(cls, data: BytesLike) -> Close:
        if isinstance(data, memoryview):
            raise AssertionError("only compressed outgoing frames use memoryview")
        if len(data) >= 2:
            (code,) = struct.unpack("!H", data[:2])
            close = cls(code, data[2:].decode())
            close.check()
            return close
        if len(data) == 0:
            return cls(CloseCode.NO_STATUS_RCVD, "")
        raise ProtocolError("close frame too short")

    def serialize(self) -> bytes:
        self.check()
        return struct.pack("!H", self.code) + self.reason.encode()

    def check(self) -> None:
        if not (self.code in EXTERNAL_CLOSE_CODES or 3000 <= self.code < 5000):
            raise ProtocolError("invalid status code")


from . import extensions

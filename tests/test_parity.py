"""Behavioral parity with websockets 16's framing implementation."""

from __future__ import annotations

import ctypes
import inspect
import random
import struct

import numpy as np
import pytest
import websockets.frames as reference
import websockets.utils as reference_utils

import mojo_websockets.frames as frames
import mojo_websockets.utils as mojo_utils
from mojo_websockets._lib import bytes_address, lib, source_buffer
from mojo_websockets.exceptions import PayloadTooBig, ProtocolError
from mojo_websockets.utils import apply_mask


def read_from(wire: bytes | bytearray):
    position = 0

    def read_exact(size: int):
        nonlocal position
        chunk = wire[position : position + size]
        if len(chunk) != size:
            raise EOFError("unexpected end of frame")
        position += size
        if False:
            yield
        return chunk

    return read_exact


def finish(generator):
    with pytest.raises(StopIteration) as stopped:
        next(generator)
    return stopped.value.value


@pytest.mark.parametrize(
    "length",
    [
        0,
        1,
        2,
        3,
        4,
        5,
        31,
        125,
        126,
        127,
        513,
        519,
        543,
        544,
        545,
        639,
        640,
        641,
        1024,
        65536,
    ],
)
def test_apply_mask_matches_upstream(length):
    data = bytes((index * 37 + 11) & 255 for index in range(length))
    mask = b"\x12\x34\x56\x78"
    assert apply_mask(data, mask) == reference.apply_mask(data, mask)
    assert apply_mask(apply_mask(data, mask), mask) == data


@pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
def test_apply_mask_accepts_bytes_like(kind):
    data = kind(b"payload")
    assert apply_mask(data, b"\x00\x01\x02\x03") == reference_utils.apply_mask(
        data, b"\x00\x01\x02\x03"
    )


def test_apply_mask_accepts_noncontiguous_memoryview():
    data = memoryview(b"0123456789")[::2]
    assert apply_mask(data, b"abcd") == reference_utils.apply_mask(data, b"abcd")


def test_apply_mask_keeps_contiguous_numpy_input_zero_copy():
    class NoBytesArray(np.ndarray):
        def __bytes__(self):
            raise AssertionError("contiguous input was copied")

    data = np.arange(1024, dtype=np.uint8).view(NoBytesArray)
    expected = reference_utils.apply_mask(data.view(np.ndarray).tobytes(), b"abcd")
    assert apply_mask(data, b"abcd") == expected


@pytest.mark.parametrize(
    "data",
    [
        np.arange(1024, dtype=np.uint16),
        np.arange(1024, dtype=np.float64).reshape(32, 32),
    ],
)
def test_apply_mask_uses_all_bytes_of_contiguous_numpy_dtypes(data):
    expected = reference_utils.apply_mask(data.tobytes(), b"abcd")
    assert apply_mask(data, b"abcd") == expected


def test_source_buffer_retains_numpy_exporter():
    data = np.arange(1024, dtype=np.uint8)
    owner, address, size = source_buffer(data)
    del data
    assert owner is not None
    assert address != 0
    assert size == 1024
    assert ctypes.string_at(address, size) == bytes(range(256)) * 4


def test_native_mask_rejects_invalid_lengths_and_null_pointers():
    destination = b"\0" * 8
    assert lib().mws_apply_mask(
        0, bytes_address(destination), 1, 1, len(destination), 0, 0
    ) == -2
    assert lib().mws_apply_mask(
        0, bytes_address(destination), 9, 8, len(destination), 0, 0
    ) == -1


def test_native_serializer_rejects_undersized_destination():
    source = b"payload"
    destination = b"\0" * 4
    assert lib().mws_serialize_frame(
        bytes_address(source),
        bytes_address(destination),
        len(source),
        len(source),
        len(destination),
        0x82,
        0,
        0,
        0,
    ) == -1


def test_apply_mask_parallel_threshold(monkeypatch):
    threshold = mojo_utils._PARALLEL_THRESHOLD
    calls = []
    initialize = mojo_utils.ensure_parallel_runtime

    def track_runtime_initialization():
        calls.append(True)
        return initialize()

    monkeypatch.setattr(
        mojo_utils,
        "ensure_parallel_runtime",
        track_runtime_initialization,
    )
    mask = b"\x12\x34\x56\x78"
    serial_data = bytes(range(256)) * ((threshold - 1) // 256) + b"x" * 255
    parallel_data = serial_data + b"tail"
    assert len(serial_data) == threshold - 1
    assert len(parallel_data) == threshold + 3
    assert apply_mask(serial_data, mask) == reference_utils.apply_mask(
        serial_data, mask
    )
    assert calls == []
    assert apply_mask(parallel_data, mask) == reference_utils.apply_mask(
        parallel_data, mask
    )
    assert calls == [True]


@pytest.mark.parametrize("mask", [b"", b"123", b"12345"])
def test_apply_mask_rejects_wrong_key_length(mask):
    with pytest.raises(ValueError, match="mask must contain 4 bytes"):
        apply_mask(b"data", mask)


@pytest.mark.parametrize("length", [0, 1, 125, 126, 127, 65535, 65536])
@pytest.mark.parametrize("opcode", [frames.OP_CONT, frames.OP_TEXT, frames.OP_BINARY])
def test_unmasked_serialization_matches_upstream(length, opcode):
    payload = bytes((index * 17) & 255 for index in range(length))
    ours = frames.Frame(opcode, payload).serialize(mask=False)
    theirs = reference.Frame(reference.Opcode(opcode), payload).serialize(mask=False)
    assert ours == theirs


@pytest.mark.parametrize("length", [0, 1, 125, 126, 65536])
def test_masked_serialization_matches_upstream(length, monkeypatch):
    key = b"\xde\xad\xbe\xef"
    monkeypatch.setattr(frames.secrets, "token_bytes", lambda size: key)
    payload = bytes((index * 29) & 255 for index in range(length))
    ours = frames.Frame(frames.OP_BINARY, payload, fin=False).serialize(mask=True)
    theirs = reference.Frame(reference.OP_BINARY, payload, fin=False).serialize(mask=True)
    assert ours == theirs


def test_large_masked_serialization_uses_parallel_path(monkeypatch):
    key = b"\xde\xad\xbe\xef"
    monkeypatch.setattr(frames.secrets, "token_bytes", lambda size: key)
    length = mojo_utils._PARALLEL_THRESHOLD + 3
    payload = bytes(range(256)) * (length // 256) + b"xyz"
    ours = frames.Frame(frames.OP_BINARY, payload).serialize(mask=True)
    theirs = reference.Frame(reference.OP_BINARY, payload).serialize(mask=True)
    assert ours == theirs


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [0, 1, 125, 126, 127, 65535, 65536])
def test_parse_matches_upstream(masked, length, monkeypatch):
    key = b"\x01\x23\x45\x67"
    monkeypatch.setattr(frames.secrets, "token_bytes", lambda size: key)
    payload = bytes((index * 13 + 7) & 255 for index in range(length))
    wire = frames.Frame(frames.OP_TEXT, payload, fin=False).serialize(mask=masked)
    ours = finish(frames.Frame.parse(read_from(wire), mask=masked))
    theirs = finish(reference.Frame.parse(read_from(wire), mask=masked))
    assert int(ours.opcode) == int(theirs.opcode)
    assert bytes(ours.data) == bytes(theirs.data)
    assert (ours.fin, ours.rsv1, ours.rsv2, ours.rsv3) == (
        theirs.fin,
        theirs.rsv1,
        theirs.rsv2,
        theirs.rsv3,
    )


def test_parser_preserves_unmasked_bytearray():
    wire = bytearray(frames.Frame(frames.OP_BINARY, b"abc").serialize(mask=False))
    parsed = finish(frames.Frame.parse(read_from(wire), mask=False))
    assert isinstance(parsed.data, bytearray)
    assert parsed.data == b"abc"


def test_parser_rejects_invalid_opcode_like_upstream():
    wire = b"\x83\x00"
    for frame_type, error in [
        (frames.Frame, ProtocolError),
        (reference.Frame, reference.ProtocolError),
    ]:
        with pytest.raises(error, match="invalid opcode"):
            finish(frame_type.parse(read_from(wire), mask=False))


def test_parser_rejects_incorrect_masking_like_upstream():
    for frame_type, error in [
        (frames.Frame, ProtocolError),
        (reference.Frame, reference.ProtocolError),
    ]:
        with pytest.raises(error, match="incorrect masking"):
            finish(frame_type.parse(read_from(b"\x82\x00"), mask=True))


def test_parser_enforces_max_size_like_upstream():
    wire = b"\x82\x7e" + struct.pack("!H", 1000)
    for frame_type, error in [
        (frames.Frame, PayloadTooBig),
        (reference.Frame, reference.PayloadTooBig),
    ]:
        with pytest.raises(error) as caught:
            finish(frame_type.parse(read_from(wire), mask=False, max_size=999))
        assert str(caught.value) == "frame with 1000 bytes exceeds limit of 999 bytes"


@pytest.mark.parametrize(
    "frame, message",
    [
        (frames.Frame(frames.OP_TEXT, b"x", rsv1=True), "reserved bits must be 0"),
        (frames.Frame(frames.OP_PING, b"x" * 126), "control frame too long"),
        (frames.Frame(frames.OP_PONG, b"x", fin=False), "fragmented control frame"),
    ],
)
def test_frame_validation(frame, message):
    with pytest.raises(ProtocolError, match=message):
        frame.serialize(mask=False)


class ReverseExtension:
    def encode(self, frame):
        return frames.Frame(
            frame.opcode,
            bytes(frame.data)[::-1],
            frame.fin,
            rsv1=True,
        )

    def decode(self, frame, *, max_size):
        assert max_size == 100
        return frames.Frame(
            frame.opcode,
            bytes(frame.data)[::-1],
            frame.fin,
            rsv1=False,
        )


def test_extension_encode_and_decode_hooks():
    extension = ReverseExtension()
    original = frames.Frame(frames.OP_TEXT, b"abcdef")
    wire = original.serialize(mask=False, extensions=[extension])
    assert wire[0] & 0x40
    decoded = finish(
        frames.Frame.parse(
            read_from(wire),
            mask=False,
            max_size=100,
            extensions=[extension],
        )
    )
    assert decoded == original


@pytest.mark.parametrize(
    "code, reason",
    [
        (1000, ""),
        (1001, "going away"),
        (3000, "registered"),
        (4999, "private"),
        (1014, "proxy"),
    ],
)
def test_close_roundtrip_matches_upstream(code, reason):
    ours = frames.Close(code, reason)
    theirs = reference.Close(code, reason)
    assert ours.serialize() == theirs.serialize()
    assert frames.Close.parse(ours.serialize()) == ours
    parsed = reference.Close.parse(theirs.serialize())
    assert (parsed.code, parsed.reason) == (ours.code, ours.reason)
    assert str(ours) == str(theirs)


def test_empty_close_payload_matches_upstream():
    ours = frames.Close.parse(b"")
    theirs = reference.Close.parse(b"")
    assert int(ours.code) == int(theirs.code) == 1005
    assert ours.reason == theirs.reason == ""


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\x03", "close frame too short"),
        (b"\x03\xed", "invalid status code"),
    ],
)
def test_invalid_close_payload(payload, message):
    with pytest.raises(ProtocolError, match=message):
        frames.Close.parse(payload)


def test_close_rejects_invalid_utf8():
    with pytest.raises(UnicodeDecodeError):
        frames.Close.parse(b"\x03\xe8\xff")


@pytest.mark.parametrize(
    "frame",
    [
        frames.Frame(frames.OP_TEXT, b"Hello"),
        frames.Frame(frames.OP_BINARY, b"Hello"),
        frames.Frame(frames.OP_TEXT, b"\xff"),
        frames.Frame(frames.OP_PING, b"ping"),
        frames.Frame(frames.OP_CONT, b"\x80tail", fin=False),
        frames.Frame(frames.OP_CLOSE, frames.Close(1000, "done").serialize()),
    ],
)
def test_frame_string_matches_upstream(frame):
    theirs = reference.Frame(
        reference.Opcode(frame.opcode),
        frame.data,
        frame.fin,
        frame.rsv1,
        frame.rsv2,
        frame.rsv3,
    )
    assert str(frame) == str(theirs)


def test_public_signatures_match_upstream():
    assert inspect.signature(frames.Frame) == inspect.signature(reference.Frame)
    assert list(inspect.signature(frames.Frame.parse).parameters) == list(
        inspect.signature(reference.Frame.parse).parameters
    )
    assert inspect.signature(frames.Frame.serialize) == inspect.signature(
        reference.Frame.serialize
    )
    assert inspect.signature(apply_mask) == inspect.signature(reference_utils.apply_mask)


def test_random_frame_corpus_matches_upstream():
    rng = random.Random(1945)
    for _ in range(250):
        opcode = rng.choice(frames.DATA_OPCODES)
        length = rng.randrange(0, 2048)
        payload = rng.randbytes(length)
        fin = rng.choice([False, True])
        ours = frames.Frame(opcode, payload, fin=fin).serialize(mask=False)
        theirs = reference.Frame(reference.Opcode(opcode), payload, fin=fin).serialize(
            mask=False
        )
        assert ours == theirs

# mojo-websockets

`mojo-websockets` is a standalone implementation of WebSocket framing and
masking with the data-plane work compiled from Mojo. Its Python API follows the
modern Sans-I/O API in `websockets.frames`, making it possible to replace imports
for the covered subset without changing call signatures.

The port targets the pieces where native code is useful: applying RFC 6455's
repeating four-byte mask and constructing complete wire frames. Python retains
the incremental reader and extension orchestration because these are I/O control
flow, not compute kernels.

## Coverage

Covered:

- `Frame`, `Opcode`, all data and control opcode constants
- generator-based `Frame.parse` and `Frame.serialize`
- masked and unmasked frames with 7-, 16-, and 64-bit payload lengths
- FIN and RSV bits, control-frame validation, payload-size limits
- extension encode and decode hooks
- `Close`, `CloseCode`, close-payload parsing, serialization, and validation
- `utils.apply_mask` and the compatible `speedups.apply_mask` import
- `ProtocolError` and `PayloadTooBig`

Not covered:

- opening handshakes, HTTP, clients, servers, or socket I/O
- protocol and connection state machines
- message reassembly or streaming messages
- implementations of extensions such as per-message deflate
- asyncio, threading, sync, or legacy APIs

This is intentionally a framing library, not a full replacement for the
upstream networking stack.

## Install and build

The repository pins the tested Mojo nightly and installs the upstream
`websockets` package for parity tests:

```bash
pixi install
pixi run build
pixi run test
```

The build writes `dist/libmojo-websockets.so`. Set `MOJO_WEBSOCKETS_LIB` to an
already-built shared library when loading the Python package from another
location.

## Usage

```python
from mojo_websockets import Frame, OP_BINARY
from mojo_websockets.utils import apply_mask

frame = Frame(OP_BINARY, b"hello")
wire = frame.serialize(mask=False)

assert wire == b"\x82\x05hello"
assert apply_mask(b"hello", b"\x01\x02\x03\x04") == b"igohn"
```

`Frame.parse`, `Frame.serialize`, `Close.parse`, `Close.serialize`, and
`apply_mask` use the same argument names and defaults as `websockets 16.1.1`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. The reference is `websockets 16.1.1` with its native C speedups
enabled.

| operation | Mojo | upstream websockets | speedup |
| --- | ---: | ---: | ---: |
| `apply_mask` 64 B | 1.31 us | 213.2 ns | 0.16x |
| `apply_mask` 4,096 B | 4.48 us | 397.5 ns | 0.09x |
| `apply_mask` 1,048,576 B | 104.09 us | 90.29 us | 0.87x |
| `apply_mask` 16,777,216 B | 942.43 us | 2.44 ms | 2.59x |
| serialize unmasked 1 MiB | 85.83 us | 95.48 us | 1.11x |
| serialize masked 1 MiB | 103.52 us | 179.38 us | 1.73x |

Small standalone mask calls are slower because the fixed ctypes transition
cost is much larger than the work. Large masks use four CPU workers; masked
frame serialization builds the header, writes the mask key, copies the payload,
and XORs it directly in the final allocation. Unmasked serialization uses
Python's optimized bytes copy. The benchmark script prints a Markdown table and
is always run under the machine-wide Pixi task lock.

There is no GPU path. WebSocket masking performs one bytewise XOR for each byte
read and written, far below the roughly two operations per byte needed to
justify host-device transfer and launch overhead. A GPU implementation would
lose, so the benchmark didn't consume shared GPU resources.

## How it works

Python owns every input and output allocation. Contiguous buffers cross ctypes
directly; writable contiguous buffer providers such as NumPy remain zero-copy
on input. The exported, non-parametric Mojo functions receive their addresses
as 64-bit integers and rebuild them as
`UnsafePointer[UInt8, AnyOrigin[mut=True]]`. No Mojo allocation crosses the FFI
boundary.

The mask kernel processes unaligned `UInt64` vectors with the four-byte key
repeated across each word, unrolls four SIMD vectors per iteration, then handles
whole-word and byte tails serially. At 4 MiB and above, `parallelize` distributes
1 MiB chunks over four CPU workers; smaller inputs remain serial. The async CPU
runtime is initialized lazily, and failure to initialize selects the serial
path. The frame kernel writes the network-order header and optional mask key
before masking the payload directly into its final immutable Python `bytes`
allocation. Payloads up to 512 bytes use Python's integer XOR path because
crossing ctypes costs more than the masking work at that size.

Incremental parsing follows the Sans-I/O generator contract: it requests exactly
the bytes needed for each header section and payload, calls the Mojo mask kernel
for substantial masked payloads, applies extension decoders in reverse order,
then performs RFC 6455 validation.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

Parity tests compare wire bytes, parsed frames, errors, extension behavior,
close payloads, public signatures, boundary lengths, and a randomized frame
corpus directly with the installed upstream package.

MIT licensed.

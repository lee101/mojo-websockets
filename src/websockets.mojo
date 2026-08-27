"""WebSocket wire kernels exported through a small C ABI."""

from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime MASK_CHUNK_SIZE = 1024 * 1024
comptime MASK_WORKERS = 4


@always_inline
def sync_parallelize[FuncType: def(Int) -> None](func: FuncType, count: Int):
    @__parameter
    @always_inline
    def wrapped(i: Int):
        func(i)

    @always_inline
    @__parameter
    async def task_fn(i: Int):
        wrapped(i)

    var tasks = TaskGroup()
    for i in range(count):
        tasks.create_task(task_fn(i))
    tasks.wait()


@always_inline
def parallelize[
    origins: OriginSet,
    //,
    func: def(Int) capturing[origins] -> None,
](num_work_items: Int, num_workers: Int):
    var workers = min(num_work_items, num_workers)
    if workers <= 1:
        for i in range(num_work_items):
            func(i)
        return

    var chunk_size, extra_items = divmod(num_work_items, workers)

    @always_inline
    def worker(worker_index: Int) {imm chunk_size, imm extra_items}:
        var start = worker_index * chunk_size + min(worker_index, extra_items)
        for i in range(chunk_size + Int(worker_index < extra_items)):
            func(start + i)

    initialize_runtime()
    sync_parallelize(worker, workers)


def bytes_ptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def copy_bytes(src: BPtr, dst: BPtr, n: Int):
    var src_words = src.unsafe_bitcast[UInt64]()
    var dst_words = dst.unsafe_bitcast[UInt64]()
    var words = n // 8
    var vector_end = words - words % W
    var word = 0
    while word < vector_end:
        dst_words.unsafe_store[alignment=1](
            word, src_words.unsafe_load[width=W, alignment=1](word)
        )
        word += W
    while word < words:
        dst_words.unsafe_store[alignment=1](
            word, src_words.unsafe_load[alignment=1](word)
        )
        word += 1
    var i = words * 8
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += 1


def mask_copy(src: BPtr, dst: BPtr, n: Int, mask: Int):
    comptime UNROLL = 4
    var src_words = src.unsafe_bitcast[UInt64]()
    var dst_words = dst.unsafe_bitcast[UInt64]()
    var repeated = UInt64(mask) | (UInt64(mask) << 32)
    var pattern = SIMD[DType.uint64, W](repeated)
    var words = n // 8
    var unrolled_end = words - words % (W * UNROLL)
    var word = 0
    while word < unrolled_end:
        dst_words.unsafe_store[alignment=1](
            word, src_words.unsafe_load[width=W, alignment=1](word) ^ pattern
        )
        dst_words.unsafe_store[alignment=1](
            word + W,
            src_words.unsafe_load[width=W, alignment=1](word + W) ^ pattern,
        )
        dst_words.unsafe_store[alignment=1](
            word + 2 * W,
            src_words.unsafe_load[width=W, alignment=1](word + 2 * W) ^ pattern,
        )
        dst_words.unsafe_store[alignment=1](
            word + 3 * W,
            src_words.unsafe_load[width=W, alignment=1](word + 3 * W) ^ pattern,
        )
        word += W * UNROLL
    var vector_end = words - words % W
    while word < vector_end:
        dst_words.unsafe_store[alignment=1](
            word, src_words.unsafe_load[width=W, alignment=1](word) ^ pattern
        )
        word += W
    while word < words:
        dst_words.unsafe_store[alignment=1](
            word, src_words.unsafe_load[alignment=1](word) ^ repeated
        )
        word += 1
    var i = words * 8
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i] ^ UInt8(
            (mask >> ((i & 3) * 8)) & 255
        )
        i += 1


def mask_copy_parallel(src: BPtr, dst: BPtr, n: Int, mask: Int):
    var chunks = (n + MASK_CHUNK_SIZE - 1) // MASK_CHUNK_SIZE

    @__parameter
    def mask_chunk(chunk: Int):
        var offset = chunk * MASK_CHUNK_SIZE
        var size = min(MASK_CHUNK_SIZE, n - offset)
        mask_copy(src.unsafe_offset(offset), dst.unsafe_offset(offset), size, mask)

    parallelize[mask_chunk](chunks, MASK_WORKERS)


@export("mws_apply_mask")
def mws_apply_mask(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    src_size: Int,
    dst_size: Int,
    mask: Int,
    use_parallel: Int,
) abi("C") -> Int:
    if n < 0 or n > src_size or n > dst_size:
        return -1
    if n == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -2
    var src = bytes_ptr(src_addr)
    var dst = bytes_ptr(dst_addr)
    if use_parallel != 0:
        mask_copy_parallel(src, dst, n, mask)
    else:
        mask_copy(src, dst, n, mask)
    return 0


@export("mws_serialize_frame")
def mws_serialize_frame(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    src_size: Int,
    dst_size: Int,
    head1: Int,
    masked: Int,
    mask: Int,
    use_parallel: Int,
) abi("C") -> Int:
    if n < 0 or n > src_size or head1 < 0 or head1 > 255:
        return -1
    var header_size = 2 if n < 126 else 4 if n < 65536 else 10
    var required = header_size + (4 if masked != 0 else 0) + n
    if required > dst_size or dst_addr == 0:
        return -1
    if n > 0 and src_addr == 0:
        return -2
    var dst = bytes_ptr(dst_addr)
    dst[unsafe_offset=0] = UInt8(head1)
    var offset = 2
    var mask_bit = 128 if masked != 0 else 0

    if n < 126:
        dst[unsafe_offset=1] = UInt8(mask_bit | n)
    elif n < 65536:
        dst[unsafe_offset=1] = UInt8(mask_bit | 126)
        dst[unsafe_offset=2] = UInt8((n >> 8) & 255)
        dst[unsafe_offset=3] = UInt8(n & 255)
        offset = 4
    else:
        dst[unsafe_offset=1] = UInt8(mask_bit | 127)
        for j in range(8):
            dst[unsafe_offset=2 + j] = UInt8((n >> ((7 - j) * 8)) & 255)
        offset = 10

    if masked != 0:
        dst[unsafe_offset=offset] = UInt8(mask & 255)
        dst[unsafe_offset=offset + 1] = UInt8((mask >> 8) & 255)
        dst[unsafe_offset=offset + 2] = UInt8((mask >> 16) & 255)
        dst[unsafe_offset=offset + 3] = UInt8((mask >> 24) & 255)
        offset += 4

    if n > 0:
        var src = bytes_ptr(src_addr)
        if masked != 0:
            if use_parallel != 0:
                mask_copy_parallel(src, dst.unsafe_offset(offset), n, mask)
            else:
                mask_copy(src, dst.unsafe_offset(offset), n, mask)
        else:
            copy_bytes(src, dst.unsafe_offset(offset), n)
    return offset + n

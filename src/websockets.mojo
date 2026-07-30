"""WebSocket wire kernels exported through a small C ABI."""

from std.algorithm import parallelize
from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def bytes_ptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def copy_bytes(src: BPtr, dst: BPtr, n: Int):
    comptime W = simd_width_of[DType.uint64]()
    var src_words = src.bitcast[UInt64]()
    var dst_words = dst.bitcast[UInt64]()
    var words = n // 8
    var vector_end = words - words % W
    var word = 0
    while word < vector_end:
        dst_words.store[alignment=1](
            word, src_words.load[width=W, alignment=1](word)
        )
        word += W
    while word < words:
        dst_words.store[alignment=1](
            word, src_words.load[alignment=1](word)
        )
        word += 1
    var i = words * 8
    while i < n:
        dst[i] = src[i]
        i += 1


def mask_copy(src: BPtr, dst: BPtr, n: Int, mask: Int):
    comptime W = simd_width_of[DType.uint64]()
    comptime UNROLL = 4
    var src_words = src.bitcast[UInt64]()
    var dst_words = dst.bitcast[UInt64]()
    var repeated = UInt64(mask) | (UInt64(mask) << 32)
    var pattern = SIMD[DType.uint64, W](repeated)
    var words = n // 8
    var unrolled_end = words - words % (W * UNROLL)
    var word = 0
    while word < unrolled_end:
        dst_words.store[alignment=1](
            word, src_words.load[width=W, alignment=1](word) ^ pattern
        )
        dst_words.store[alignment=1](
            word + W,
            src_words.load[width=W, alignment=1](word + W) ^ pattern,
        )
        dst_words.store[alignment=1](
            word + 2 * W,
            src_words.load[width=W, alignment=1](word + 2 * W) ^ pattern,
        )
        dst_words.store[alignment=1](
            word + 3 * W,
            src_words.load[width=W, alignment=1](word + 3 * W) ^ pattern,
        )
        word += W * UNROLL
    var vector_end = words - words % W
    while word < vector_end:
        dst_words.store[alignment=1](
            word, src_words.load[width=W, alignment=1](word) ^ pattern
        )
        word += W
    while word < words:
        dst_words.store[alignment=1](
            word, src_words.load[alignment=1](word) ^ repeated
        )
        word += 1
    var i = words * 8
    while i < n:
        dst[i] = src[i] ^ UInt8((mask >> ((i & 3) * 8)) & 255)
        i += 1


def mask_copy_parallel(src: BPtr, dst: BPtr, n: Int, mask: Int):
    comptime CHUNK_SIZE = 1024 * 1024
    var num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    @parameter
    def apply_chunk(chunk: Int):
        var offset = chunk * CHUNK_SIZE
        var chunk_size = min(CHUNK_SIZE, n - offset)
        mask_copy(src + offset, dst + offset, chunk_size, mask)

    parallelize[apply_chunk](num_chunks, 4)


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
    dst[0] = UInt8(head1)
    var offset = 2
    var mask_bit = 128 if masked != 0 else 0

    if n < 126:
        dst[1] = UInt8(mask_bit | n)
    elif n < 65536:
        dst[1] = UInt8(mask_bit | 126)
        dst[2] = UInt8((n >> 8) & 255)
        dst[3] = UInt8(n & 255)
        offset = 4
    else:
        dst[1] = UInt8(mask_bit | 127)
        for j in range(8):
            dst[2 + j] = UInt8((n >> ((7 - j) * 8)) & 255)
        offset = 10

    if masked != 0:
        dst[offset] = UInt8(mask & 255)
        dst[offset + 1] = UInt8((mask >> 8) & 255)
        dst[offset + 2] = UInt8((mask >> 16) & 255)
        dst[offset + 3] = UInt8((mask >> 24) & 255)
        offset += 4

    if n > 0:
        var src = bytes_ptr(src_addr)
        if masked != 0:
            if use_parallel != 0:
                mask_copy_parallel(src, dst + offset, n, mask)
            else:
                mask_copy(src, dst + offset, n, mask)
        else:
            copy_bytes(src, dst + offset, n)
    return offset + n

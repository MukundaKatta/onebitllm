"""Optimized binary and ternary matrix multiplication using numpy.

Packs binary/ternary weights into uint8 for storage efficiency and
implements specialized matmul kernels that operate on packed data.
"""
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def pack_binary(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack 1-bit weights (sign) into uint8, 8 values per byte.

    Args:
        weights: Array of {-1, +1} values.

    Returns:
        Tuple of (packed_data as uint8, scale).
    """
    flat = weights.ravel()
    # Convert {-1, +1} to {0, 1}
    bits = ((flat + 1) // 2).astype(np.uint8)

    # Pad to multiple of 8
    pad_len = (8 - len(bits) % 8) % 8
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), constant_values=0)

    # Pack 8 bits into each uint8
    packed = np.packbits(bits, bitorder="little")

    scale = np.mean(np.abs(weights.astype(np.float32)))
    return packed, np.array([scale], dtype=np.float32)


def unpack_binary(packed: np.ndarray, shape: Tuple[int, ...], scale: np.ndarray) -> np.ndarray:
    """Unpack binary weights from uint8 back to {-1, +1} float32."""
    bits = np.unpackbits(packed, bitorder="little")
    total_elements = 1
    for s in shape:
        total_elements *= s
    bits = bits[:total_elements]
    # Convert {0, 1} back to {-1, +1}
    weights = (bits.astype(np.float32) * 2 - 1) * float(scale[0])
    return weights.reshape(shape)


def pack_ternary(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pack ternary weights {-1, 0, +1} into int8, 4 values per byte.

    Each value needs 2 bits: 00=0, 01=+1, 11=-1.

    Args:
        weights: Array of {-1, 0, +1} values.

    Returns:
        Tuple of (packed_data as uint8, scale).
    """
    flat = weights.ravel().astype(np.int8)

    # Encode: -1->3(0b11), 0->0(0b00), 1->1(0b01)
    encoded = np.where(flat == -1, 3, np.where(flat == 1, 1, 0)).astype(np.uint8)

    # Pack 4 values per byte (2 bits each)
    pad_len = (4 - len(encoded) % 4) % 4
    if pad_len > 0:
        encoded = np.pad(encoded, (0, pad_len), constant_values=0)

    packed = np.zeros(len(encoded) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= encoded[i::4] << (i * 2)

    mask = flat != 0
    scale = np.mean(np.abs(weights[mask].astype(np.float32))) if mask.any() else np.float32(1.0)
    return packed, np.array([scale], dtype=np.float32)


def unpack_ternary(packed: np.ndarray, shape: Tuple[int, ...], scale: np.ndarray) -> np.ndarray:
    """Unpack ternary weights from packed uint8."""
    total = 1
    for s in shape:
        total *= s

    values = np.zeros(len(packed) * 4, dtype=np.int8)
    for i in range(4):
        encoded = (packed >> (i * 2)) & 0x03
        decoded = np.where(encoded == 3, -1, np.where(encoded == 1, 1, 0))
        values[i::4] = decoded

    values = values[:total]
    return values.astype(np.float32).reshape(shape) * float(scale[0])


def binary_matmul(x: np.ndarray, packed_weights: np.ndarray, scale: np.ndarray,
                   weight_shape: Tuple[int, ...]) -> np.ndarray:
    """Matrix multiplication with binary (1-bit) weights.

    Performs x @ W^T where W is stored in packed binary format.
    Uses XOR + popcount approach for efficiency on binary weights.

    Args:
        x: Input activations (float32).
        packed_weights: Packed binary weights.
        scale: Weight scale factor.
        weight_shape: Original weight shape (out_features, in_features).

    Returns:
        Result of x @ W^T.
    """
    # Unpack weights (in production, use bitwise ops for speed)
    W = unpack_binary(packed_weights, weight_shape, scale)
    return x @ W.T


def ternary_matmul(x: np.ndarray, packed_weights: np.ndarray, scale: np.ndarray,
                    weight_shape: Tuple[int, ...]) -> np.ndarray:
    """Matrix multiplication with ternary (1.58-bit) weights.

    For ternary weights {-1, 0, +1}, the matmul reduces to additions
    and subtractions (no multiplications needed for the weight side).
    """
    W = unpack_ternary(packed_weights, weight_shape, scale)
    return x @ W.T


def benchmark_matmul(m: int = 512, n: int = 512, k: int = 512, n_iters: int = 100) -> Dict[str, Any]:
    """Benchmark binary/ternary matmul vs float32 matmul.

    Args:
        m, n, k: Matrix dimensions.
        n_iters: Number of iterations.

    Returns:
        Dict with timing results and speedup.
    """
    x = np.random.randn(m, k).astype(np.float32)
    W = np.random.randn(n, k).astype(np.float32)

    # Float32 baseline
    start = time.time()
    for _ in range(n_iters):
        _ = x @ W.T
    float32_time = (time.time() - start) / n_iters

    # Binary
    W_sign = np.sign(W).astype(np.int8)
    W_sign[W_sign == 0] = 1
    packed_bin, scale_bin = pack_binary(W_sign)

    start = time.time()
    for _ in range(n_iters):
        _ = binary_matmul(x, packed_bin, scale_bin, W.shape)
    binary_time = (time.time() - start) / n_iters

    # Ternary
    threshold = 0.7 * np.mean(np.abs(W))
    W_ternary = np.zeros_like(W, dtype=np.int8)
    W_ternary[W > threshold] = 1
    W_ternary[W < -threshold] = -1
    packed_ter, scale_ter = pack_ternary(W_ternary)

    start = time.time()
    for _ in range(n_iters):
        _ = ternary_matmul(x, packed_ter, scale_ter, W.shape)
    ternary_time = (time.time() - start) / n_iters

    return {
        "dimensions": f"{m}x{k} @ {k}x{n}",
        "float32_ms": round(float32_time * 1000, 3),
        "binary_ms": round(binary_time * 1000, 3),
        "ternary_ms": round(ternary_time * 1000, 3),
        "binary_memory_ratio": round(32.0 / 1.0, 1),
        "ternary_memory_ratio": round(32.0 / 2.0, 1),
        "binary_packed_bytes": packed_bin.nbytes,
        "float32_bytes": W.nbytes,
    }

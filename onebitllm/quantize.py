"""Ultra-low-bit quantization: 1-bit, 1.58-bit (ternary), 2-bit, 4-bit.

Implements real quantization math with scale factors for compressing
neural network weights to extreme low-bit representations.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QuantizedWeight:
    """A quantized weight tensor."""
    data: np.ndarray
    scale: np.ndarray
    zero_point: Optional[np.ndarray] = None
    bits: float = 1.0
    original_shape: Tuple[int, ...] = ()
    method: str = "sign"


def quantize_1bit(weights: np.ndarray) -> QuantizedWeight:
    """1-bit quantization: sign function.

    Maps each weight to {-1, +1} based on its sign.
    Scale factor = mean(|w|).

    W_q = sign(W), scale = mean(|W|)
    W_hat = scale * W_q

    Args:
        weights: Float weight tensor.

    Returns:
        QuantizedWeight with binary values.
    """
    scale = np.mean(np.abs(weights), axis=-1, keepdims=True)
    scale = np.clip(scale, 1e-8, None)

    # Sign quantization: 0 maps to +1
    quantized = np.where(weights >= 0, np.int8(1), np.int8(-1))

    return QuantizedWeight(
        data=quantized, scale=scale.squeeze(), bits=1.0,
        original_shape=weights.shape, method="sign",
    )


def quantize_ternary(weights: np.ndarray, threshold_factor: float = 0.7) -> QuantizedWeight:
    """1.58-bit (ternary) quantization: maps to {-1, 0, +1}.

    Uses a threshold based on the mean absolute value to determine
    which weights are zeroed out.

    W_q[i] = sign(W[i]) if |W[i]| > threshold else 0
    threshold = factor * mean(|W|)

    Args:
        weights: Float weight tensor.
        threshold_factor: Multiplier for the threshold (0.7 is typical).

    Returns:
        QuantizedWeight with ternary values.
    """
    mean_abs = np.mean(np.abs(weights), axis=-1, keepdims=True)
    threshold = threshold_factor * mean_abs

    quantized = np.zeros_like(weights, dtype=np.int8)
    quantized[weights > threshold] = 1
    quantized[weights < -threshold] = -1

    # Scale: mean of absolute values of non-zero weights
    mask = quantized != 0
    if mask.any():
        scale = np.mean(np.abs(weights[mask]))
    else:
        scale = float(np.mean(np.abs(weights)))

    return QuantizedWeight(
        data=quantized, scale=np.array([scale]), bits=1.58,
        original_shape=weights.shape, method="ternary",
    )


def quantize_2bit(weights: np.ndarray) -> QuantizedWeight:
    """2-bit quantization: maps to {-1, -1/3, 1/3, 1}.

    Symmetric 2-bit quantization with 4 levels.

    Args:
        weights: Float weight tensor.

    Returns:
        QuantizedWeight with 2-bit values stored as int8.
    """
    max_abs = np.max(np.abs(weights), axis=-1, keepdims=True)
    max_abs = np.clip(max_abs, 1e-8, None)
    scale = max_abs

    # Normalize to [-1, 1]
    normalized = weights / scale

    # Map to 4 levels: -1, -1/3, 1/3, 1 -> encoded as -2, -1, 1, 2
    quantized = np.zeros_like(weights, dtype=np.int8)
    quantized[normalized >= 2 / 3] = 2
    quantized[(normalized >= 0) & (normalized < 2 / 3)] = 1
    quantized[(normalized < 0) & (normalized > -2 / 3)] = -1
    quantized[normalized <= -2 / 3] = -2

    return QuantizedWeight(
        data=quantized, scale=scale.squeeze(), bits=2.0,
        original_shape=weights.shape, method="2bit",
    )


def quantize_4bit(weights: np.ndarray, group_size: int = 128) -> QuantizedWeight:
    """4-bit quantization with group-wise scaling.

    Divides the weight tensor into groups along the last axis and
    applies independent scaling per group for better accuracy.

    Args:
        weights: Float weight tensor.
        group_size: Number of elements per quantization group.

    Returns:
        QuantizedWeight with 4-bit values (stored as int8, range [-8, 7]).
    """
    shape = weights.shape
    flat = weights.reshape(-1, shape[-1]) if weights.ndim > 1 else weights.reshape(1, -1)
    rows, cols = flat.shape

    # Pad columns to be divisible by group_size
    pad = (group_size - cols % group_size) % group_size
    if pad > 0:
        flat = np.pad(flat, ((0, 0), (0, pad)), constant_values=0)

    n_groups = flat.shape[1] // group_size
    grouped = flat.reshape(rows, n_groups, group_size)

    # Per-group scale
    max_abs = np.max(np.abs(grouped), axis=-1, keepdims=True)
    max_abs = np.clip(max_abs, 1e-8, None)
    scale = max_abs / 7.0  # 4-bit signed: [-8, 7]

    quantized = np.clip(np.round(grouped / scale), -8, 7).astype(np.int8)

    return QuantizedWeight(
        data=quantized, scale=scale.squeeze(),
        bits=4.0, original_shape=shape, method="4bit_grouped",
    )


def dequantize(qw: QuantizedWeight) -> np.ndarray:
    """Dequantize a weight tensor back to float32.

    Args:
        qw: Quantized weight.

    Returns:
        Reconstructed float32 tensor.
    """
    if qw.method == "sign":
        scale = qw.scale
        if scale.ndim == 0:
            return qw.data.astype(np.float32) * float(scale)
        return qw.data.astype(np.float32) * scale.reshape(-1, 1) if qw.data.ndim > 1 else qw.data.astype(np.float32) * scale

    elif qw.method == "ternary":
        return qw.data.astype(np.float32) * float(qw.scale[0])

    elif qw.method == "2bit":
        # Decode: -2->-1, -1->-1/3, 1->1/3, 2->1
        decode_map = {-2: -1.0, -1: -1 / 3, 0: 0.0, 1: 1 / 3, 2: 1.0}
        decoded = np.vectorize(lambda x: decode_map.get(x, 0.0))(qw.data).astype(np.float32)
        scale = qw.scale
        if scale.ndim == 0:
            return decoded * float(scale)
        return decoded * scale.reshape(-1, 1) if decoded.ndim > 1 else decoded * scale

    elif qw.method == "4bit_grouped":
        scale = qw.scale
        if qw.data.ndim == 3:
            # (rows, n_groups, group_size) * (rows, n_groups, 1)
            if scale.ndim == 2:
                result = qw.data.astype(np.float32) * scale[:, :, np.newaxis]
            else:
                result = qw.data.astype(np.float32) * float(scale)
            rows, n_groups, group_size = qw.data.shape
            result = result.reshape(rows, -1)
            # Trim to original shape
            orig_cols = qw.original_shape[-1] if len(qw.original_shape) > 1 else qw.original_shape[0]
            return result[:, :orig_cols].reshape(qw.original_shape)
        return qw.data.astype(np.float32) * scale

    raise ValueError(f"Unknown quantization method: {qw.method}")


class BitQuantizer:
    """Unified quantizer supporting 1-bit, 1.58-bit, 2-bit, and 4-bit.

    Example:
        quantizer = BitQuantizer(bits=1.58)
        qw = quantizer.quantize(weight_matrix)
        reconstructed = quantizer.dequantize(qw)
        error = quantizer.compute_error(weight_matrix, qw)
    """

    def __init__(self, bits: float = 1.0, group_size: int = 128, threshold_factor: float = 0.7):
        self.bits = bits
        self.group_size = group_size
        self.threshold_factor = threshold_factor

    def quantize(self, weights: np.ndarray) -> QuantizedWeight:
        if self.bits == 1.0:
            return quantize_1bit(weights)
        elif self.bits == 1.58:
            return quantize_ternary(weights, self.threshold_factor)
        elif self.bits == 2.0:
            return quantize_2bit(weights)
        elif self.bits == 4.0:
            return quantize_4bit(weights, self.group_size)
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

    def dequantize(self, qw: QuantizedWeight) -> np.ndarray:
        return dequantize(qw)

    def compute_error(self, original: np.ndarray, qw: QuantizedWeight) -> Dict[str, float]:
        reconstructed = self.dequantize(qw)
        if reconstructed.shape != original.shape:
            reconstructed = reconstructed.reshape(original.shape) if reconstructed.size == original.size else reconstructed[:original.shape[0], :original.shape[1]]
        mse = float(np.mean((original - reconstructed) ** 2))
        mae = float(np.mean(np.abs(original - reconstructed)))
        snr = float(10 * np.log10(np.var(original) / max(mse, 1e-10)))
        compression = 32.0 / self.bits
        return {"mse": round(mse, 8), "mae": round(mae, 6), "snr_db": round(snr, 2),
                "compression_ratio": round(compression, 2), "bits": self.bits}

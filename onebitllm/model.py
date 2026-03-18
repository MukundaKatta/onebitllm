"""BitLinear layer and BitNet model: quantized weights with full-precision activations.

Implements the BitNet architecture where weight matrices use ultra-low-bit
quantization while activations remain in full precision (float32/float16).
"""
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .quantize import BitQuantizer, QuantizedWeight, dequantize

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a BitNet transformer model."""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    bits: float = 1.58
    group_size: int = 128


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMS Layer Normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class BitLinear:
    """Linear layer with quantized weights and full-precision activations.

    The weight matrix W is quantized to ultra-low-bit representation, but
    input activations are kept in float32. During forward pass, weights are
    dequantized just-in-time for the matmul.

    Optionally applies activation quantization (AbsMax) as in BitNet b1.58.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Weight quantization bits.
        quantize_activations: Whether to quantize activations too.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: float = 1.58,
        quantize_activations: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.quantize_activations = quantize_activations

        self.quantizer = BitQuantizer(bits=bits)

        # Initialize with Kaiming normal, then quantize
        scale = math.sqrt(2.0 / in_features)
        self.original_weight = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.bias = np.zeros(out_features, dtype=np.float32)

        # Layer norm parameters for activation quantization
        self.ln_weight = np.ones(in_features, dtype=np.float32)
        self.ln_bias = np.zeros(in_features, dtype=np.float32)

        self._quantize()

    def _quantize(self) -> None:
        """Quantize the weight matrix."""
        self.quantized_weight = self.quantizer.quantize(self.original_weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: LayerNorm -> (optional activation quant) -> dequantized matmul.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Layer normalization before quantized matmul (BitNet style)
        x_norm = _layer_norm(x, self.ln_weight, self.ln_bias)

        # Optional activation quantization (AbsMax)
        if self.quantize_activations:
            gamma = np.max(np.abs(x_norm), axis=-1, keepdims=True)
            gamma = np.clip(gamma, 1e-8, None)
            Qb = 2 ** (8 - 1) - 1  # 127 for 8-bit
            x_quant = np.clip(np.round(x_norm * Qb / gamma), -Qb, Qb)
            x_norm = x_quant * gamma / Qb

        # Dequantize weights and compute matmul
        W = dequantize(self.quantized_weight)
        y = x_norm @ W.T + self.bias

        return y

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage of quantized weights."""
        n_elements = self.out_features * self.in_features
        return int(n_elements * self.bits / 8) + self.quantized_weight.scale.nbytes

    @property
    def compression_ratio(self) -> float:
        return 32.0 / self.bits


class BitNetBlock:
    """A single transformer block with BitLinear layers.

    Implements self-attention + MLP with quantized weight matrices.
    """

    def __init__(self, config: ModelConfig):
        d = config.hidden_size
        h = config.num_attention_heads
        assert d % h == 0
        self.head_dim = d // h
        self.n_heads = h

        self.q_proj = BitLinear(d, d, bits=config.bits)
        self.k_proj = BitLinear(d, d, bits=config.bits)
        self.v_proj = BitLinear(d, d, bits=config.bits)
        self.o_proj = BitLinear(d, d, bits=config.bits)

        self.gate_proj = BitLinear(d, config.intermediate_size, bits=config.bits)
        self.up_proj = BitLinear(d, config.intermediate_size, bits=config.bits)
        self.down_proj = BitLinear(config.intermediate_size, d, bits=config.bits)

        self.input_ln_weight = np.ones(d, dtype=np.float32)
        self.input_ln_bias = np.zeros(d, dtype=np.float32)
        self.post_attn_ln_weight = np.ones(d, dtype=np.float32)
        self.post_attn_ln_bias = np.zeros(d, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the transformer block.

        Args:
            x: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Output of same shape.
        """
        batch, seq_len, d = x.shape

        # Self-attention
        residual = x
        x_norm = _layer_norm(x, self.input_ln_weight, self.input_ln_bias)

        q = self.q_proj.forward(x_norm).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj.forward(x_norm).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj.forward(x_norm).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        scores += mask

        attn_weights = _softmax(scores, axis=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        x = residual + self.o_proj.forward(attn_output)

        # MLP (SwiGLU)
        residual = x
        x_norm = _layer_norm(x, self.post_attn_ln_weight, self.post_attn_ln_bias)
        gate = _gelu(self.gate_proj.forward(x_norm))
        up = self.up_proj.forward(x_norm)
        x = residual + self.down_proj.forward(gate * up)

        return x


class BitNetModel:
    """Complete BitNet transformer model with quantized weights.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.embed = np.random.randn(config.vocab_size, config.hidden_size).astype(np.float32) * 0.02
        self.blocks = [BitNetBlock(config) for _ in range(config.num_hidden_layers)]
        self.final_ln_weight = np.ones(config.hidden_size, dtype=np.float32)
        self.final_ln_bias = np.zeros(config.hidden_size, dtype=np.float32)
        self.lm_head = np.random.randn(config.vocab_size, config.hidden_size).astype(np.float32) * 0.02

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            input_ids: (batch, seq_len) integer token IDs.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        x = self.embed[input_ids]
        for block in self.blocks:
            x = block.forward(x)
        x = _layer_norm(x, self.final_ln_weight, self.final_ln_bias)
        logits = x @ self.lm_head.T
        return logits

    def get_stats(self) -> Dict[str, Any]:
        total_params = self.embed.size + self.lm_head.size
        quantized_params = 0
        for block in self.blocks:
            for layer in [block.q_proj, block.k_proj, block.v_proj, block.o_proj,
                          block.gate_proj, block.up_proj, block.down_proj]:
                total_params += layer.in_features * layer.out_features
                quantized_params += layer.in_features * layer.out_features

        return {
            "n_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "total_params": total_params,
            "quantized_params": quantized_params,
            "bits": self.config.bits,
            "theoretical_compression": round(32.0 / self.config.bits, 2),
        }

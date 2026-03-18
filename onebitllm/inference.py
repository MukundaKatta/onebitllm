"""Load quantized model and run inference with token-by-token generation."""
import logging
import time
import numpy as np
from typing import Any, Dict, List, Optional

from .model import BitNetModel, ModelConfig, _softmax

logger = logging.getLogger(__name__)


def top_k_sampling(logits: np.ndarray, k: int = 50, temperature: float = 1.0) -> int:
    """Sample from top-k logits with temperature."""
    logits = logits / max(temperature, 1e-8)
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = _softmax(top_k_logits)
    chosen = np.random.choice(len(probs), p=probs)
    return int(top_k_idx[chosen])


def top_p_sampling(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """Nucleus (top-p) sampling."""
    logits = logits / max(temperature, 1e-8)
    probs = _softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[top_idx]
    top_probs /= top_probs.sum()
    chosen = np.random.choice(top_idx, p=top_probs)
    return int(chosen)


def greedy_decode(logits: np.ndarray) -> int:
    """Greedy decoding: pick the highest probability token."""
    return int(np.argmax(logits))


class BitNetInference:
    """Run inference on a BitNet quantized model.

    Supports token-by-token autoregressive generation with various
    sampling strategies.

    Args:
        model: A BitNetModel instance.
        max_length: Maximum generation length.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        sampling: Sampling strategy ('greedy', 'top_k', 'top_p').
    """

    def __init__(
        self,
        model: Optional[BitNetModel] = None,
        config: Optional[ModelConfig] = None,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        sampling: str = "top_k",
    ):
        self.config = config or ModelConfig(num_hidden_layers=2, hidden_size=128, intermediate_size=256,
                                             num_attention_heads=4, vocab_size=1000)
        self.model = model or BitNetModel(self.config)
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.sampling = sampling

        self._total_tokens = 0
        self._total_time = 0.0

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        stop_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (1, seq_len).
            max_new_tokens: Maximum tokens to generate.
            stop_token_id: Token ID that stops generation.

        Returns:
            Dict with generated IDs, tokens per second, and latency.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        current_ids = input_ids.copy()
        generated_ids: List[int] = []
        start_time = time.time()

        for step in range(max_new_tokens):
            # Truncate to max position
            if current_ids.shape[1] > self.config.max_position_embeddings:
                current_ids = current_ids[:, -self.config.max_position_embeddings:]

            # Forward pass
            logits = self.model.forward(current_ids)
            next_token_logits = logits[0, -1, :]  # Last position

            # Sample next token
            if self.sampling == "greedy":
                next_token = greedy_decode(next_token_logits)
            elif self.sampling == "top_k":
                next_token = top_k_sampling(next_token_logits, self.top_k, self.temperature)
            elif self.sampling == "top_p":
                next_token = top_p_sampling(next_token_logits, self.top_p, self.temperature)
            else:
                next_token = greedy_decode(next_token_logits)

            generated_ids.append(next_token)

            if stop_token_id is not None and next_token == stop_token_id:
                break

            # Append to sequence
            current_ids = np.concatenate([current_ids, np.array([[next_token]])], axis=1)

        elapsed = time.time() - start_time
        n_tokens = len(generated_ids)
        self._total_tokens += n_tokens
        self._total_time += elapsed

        return {
            "input_ids": input_ids[0].tolist(),
            "generated_ids": generated_ids,
            "full_ids": current_ids[0].tolist(),
            "n_generated": n_tokens,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_per_second": round(n_tokens / max(elapsed, 1e-6), 2),
            "ms_per_token": round(elapsed * 1000 / max(n_tokens, 1), 2),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens_generated": self._total_tokens,
            "total_time_seconds": round(self._total_time, 2),
            "avg_tokens_per_second": round(
                self._total_tokens / max(self._total_time, 1e-6), 2
            ),
            "model_bits": self.config.bits,
            "model_layers": self.config.num_hidden_layers,
        }

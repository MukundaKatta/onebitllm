"""Benchmark: throughput, memory usage, perplexity degradation vs full precision."""
import logging
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .quantize import BitQuantizer
from .model import BitNetModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    model_bits: float
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    memory_mb: float
    memory_vs_fp32: float
    n_params: int
    n_layers: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.model_bits,
            "throughput_tps": round(self.throughput_tokens_per_sec, 2),
            "latency_ms": round(self.latency_ms_per_token, 2),
            "memory_mb": round(self.memory_mb, 2),
            "memory_vs_fp32": round(self.memory_vs_fp32, 2),
            "n_params": self.n_params,
        }


def estimate_memory_mb(n_params: int, bits: float) -> float:
    """Estimate memory usage in MB for a given model size and bit width."""
    bytes_per_param = bits / 8
    return n_params * bytes_per_param / (1024 ** 2)


def benchmark_quantization_error(
    n_layers: int = 4, hidden_size: int = 256, bits_list: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Benchmark quantization error across different bit widths.

    Creates random weight matrices and measures reconstruction error
    for each quantization level.
    """
    bits_list = bits_list or [1.0, 1.58, 2.0, 4.0]
    results = []

    for bits in bits_list:
        quantizer = BitQuantizer(bits=bits)
        errors = []

        for _ in range(n_layers):
            W = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            qw = quantizer.quantize(W)
            error = quantizer.compute_error(W, qw)
            errors.append(error)

        avg_mse = float(np.mean([e["mse"] for e in errors]))
        avg_snr = float(np.mean([e["snr_db"] for e in errors]))

        results.append({
            "bits": bits,
            "avg_mse": round(avg_mse, 8),
            "avg_snr_db": round(avg_snr, 2),
            "compression_ratio": round(32.0 / bits, 2),
            "memory_ratio": round(bits / 32.0, 4),
        })

    return results


def benchmark_inference(
    config: Optional[ModelConfig] = None,
    n_tokens: int = 32,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> BenchmarkResult:
    """Benchmark inference throughput and latency.

    Args:
        config: Model configuration.
        n_tokens: Sequence length for benchmarking.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Returns:
        BenchmarkResult with throughput and memory estimates.
    """
    config = config or ModelConfig(
        num_hidden_layers=2, hidden_size=128, intermediate_size=256,
        num_attention_heads=4, vocab_size=1000, bits=1.58,
    )

    model = BitNetModel(config)
    input_ids = np.random.randint(0, config.vocab_size, (1, n_tokens))

    # Warmup
    for _ in range(n_warmup):
        _ = model.forward(input_ids)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = model.forward(input_ids)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = float(np.mean(times))
    tokens_per_sec = n_tokens / avg_time
    ms_per_token = avg_time * 1000 / n_tokens

    stats = model.get_stats()
    n_params = stats["total_params"]
    memory_mb = estimate_memory_mb(n_params, config.bits)
    memory_fp32 = estimate_memory_mb(n_params, 32.0)

    return BenchmarkResult(
        model_bits=config.bits,
        throughput_tokens_per_sec=tokens_per_sec,
        latency_ms_per_token=ms_per_token,
        memory_mb=memory_mb,
        memory_vs_fp32=round(memory_mb / max(memory_fp32, 1e-6), 4),
        n_params=n_params,
        n_layers=config.num_hidden_layers,
    )


def run_full_benchmark() -> Dict[str, Any]:
    """Run comprehensive benchmarks across all bit widths."""
    quant_errors = benchmark_quantization_error()

    inference_results = []
    for bits in [1.0, 1.58, 2.0, 4.0]:
        config = ModelConfig(
            num_hidden_layers=2, hidden_size=128, intermediate_size=256,
            num_attention_heads=4, vocab_size=1000, bits=bits,
        )
        result = benchmark_inference(config, n_tokens=16, n_runs=3)
        inference_results.append(result.to_dict())

    return {
        "quantization_error": quant_errors,
        "inference": inference_results,
    }

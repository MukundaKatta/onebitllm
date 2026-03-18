"""Tests for binary matmul correctness vs numpy."""
import numpy as np
import pytest
from onebitllm.kernels import (
    pack_binary, unpack_binary, pack_ternary, unpack_ternary,
    binary_matmul, ternary_matmul, benchmark_matmul,
)


class TestBinaryPacking:
    def test_pack_unpack_roundtrip(self):
        W = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1], dtype=np.int8)
        packed, scale = pack_binary(W)
        unpacked = unpack_binary(packed, W.shape, scale)
        signs = np.sign(unpacked).astype(np.int8)
        np.testing.assert_array_equal(signs, W)

    def test_packing_reduces_size(self):
        W = np.random.choice([-1, 1], size=1024).astype(np.int8)
        packed, _ = pack_binary(W)
        assert packed.nbytes <= W.nbytes // 7  # At least 7x compression (8 bits -> 1)


class TestTernaryPacking:
    def test_pack_unpack_roundtrip(self):
        W = np.array([1, 0, -1, 0, 1, 1, -1, -1], dtype=np.int8)
        packed, scale = pack_ternary(W)
        unpacked = unpack_ternary(packed, W.shape, scale)
        signs = np.sign(unpacked).astype(np.int8)
        np.testing.assert_array_equal(signs, W)

    def test_zero_preservation(self):
        W = np.array([0, 0, 0, 0], dtype=np.int8)
        packed, scale = pack_ternary(W)
        unpacked = unpack_ternary(packed, W.shape, scale)
        np.testing.assert_array_equal(np.sign(unpacked).astype(np.int8), W)


class TestBinaryMatmul:
    def test_correctness(self):
        np.random.seed(42)
        x = np.random.randn(4, 16).astype(np.float32)
        W = np.random.choice([-1, 1], size=(8, 16)).astype(np.int8)
        scale = np.array([1.0], dtype=np.float32)
        packed, _ = pack_binary(W)

        result = binary_matmul(x, packed, scale, W.shape)
        expected = x @ (W.astype(np.float32) * float(scale)).T
        np.testing.assert_allclose(result, expected, atol=1e-4)


class TestTernaryMatmul:
    def test_correctness(self):
        np.random.seed(42)
        x = np.random.randn(4, 16).astype(np.float32)
        W = np.random.choice([-1, 0, 1], size=(8, 16)).astype(np.int8)
        packed, scale = pack_ternary(W)

        result = ternary_matmul(x, packed, scale, W.shape)
        expected = x @ (W.astype(np.float32) * float(scale[0])).T
        np.testing.assert_allclose(result, expected, atol=1e-4)


class TestBenchmark:
    def test_benchmark_runs(self):
        result = benchmark_matmul(m=32, n=32, k=32, n_iters=5)
        assert "float32_ms" in result
        assert "binary_ms" in result
        assert result["binary_memory_ratio"] > 1

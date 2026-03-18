"""Tests for quantization: round-trip error bounds."""
import numpy as np
import pytest
from onebitllm.quantize import BitQuantizer, quantize_1bit, quantize_ternary, quantize_2bit, quantize_4bit, dequantize


class TestBitQuantizer:
    def test_1bit_roundtrip(self):
        W = np.random.randn(64, 32).astype(np.float32)
        quantizer = BitQuantizer(bits=1.0)
        qw = quantizer.quantize(W)
        reconstructed = quantizer.dequantize(qw)
        assert reconstructed.shape == W.shape
        error = quantizer.compute_error(W, qw)
        assert error["snr_db"] > 0

    def test_ternary_roundtrip(self):
        W = np.random.randn(64, 32).astype(np.float32)
        quantizer = BitQuantizer(bits=1.58)
        qw = quantizer.quantize(W)
        reconstructed = quantizer.dequantize(qw)
        assert reconstructed.shape == W.shape
        # Ternary values should be {-1, 0, 1} * scale
        unique = np.unique(qw.data)
        assert all(v in [-1, 0, 1] for v in unique)

    def test_2bit_roundtrip(self):
        W = np.random.randn(32, 16).astype(np.float32)
        quantizer = BitQuantizer(bits=2.0)
        qw = quantizer.quantize(W)
        reconstructed = quantizer.dequantize(qw)
        assert reconstructed.shape == W.shape

    def test_4bit_roundtrip(self):
        W = np.random.randn(64, 128).astype(np.float32)
        quantizer = BitQuantizer(bits=4.0)
        qw = quantizer.quantize(W)
        reconstructed = quantizer.dequantize(qw)
        error = quantizer.compute_error(W, qw)
        # 4-bit should have much better SNR than 1-bit
        assert error["snr_db"] > 10

    def test_error_decreases_with_bits(self):
        W = np.random.randn(64, 64).astype(np.float32) * 0.1
        errors = {}
        for bits in [1.0, 1.58, 2.0, 4.0]:
            q = BitQuantizer(bits=bits)
            qw = q.quantize(W)
            errors[bits] = q.compute_error(W, qw)["mse"]
        assert errors[4.0] < errors[1.0]

    def test_1bit_values(self):
        W = np.array([[1.0, -2.0, 3.0], [-0.5, 0.5, -1.5]]).astype(np.float32)
        qw = quantize_1bit(W)
        assert set(np.unique(qw.data)) == {-1, 1}

    def test_compression_ratio(self):
        quantizer = BitQuantizer(bits=1.0)
        W = np.random.randn(100, 100).astype(np.float32)
        qw = quantizer.quantize(W)
        error = quantizer.compute_error(W, qw)
        assert error["compression_ratio"] == 32.0


class TestQuantize1bit:
    def test_sign_quantization(self):
        W = np.array([1.0, -1.0, 0.5, -0.5, 0.0]).astype(np.float32)
        qw = quantize_1bit(W)
        assert qw.data[0] == 1
        assert qw.data[1] == -1
        assert qw.data[4] == 1  # 0 maps to +1

    def test_scale_computation(self):
        W = np.array([[2.0, -4.0], [1.0, -3.0]]).astype(np.float32)
        qw = quantize_1bit(W)
        expected_scale = np.mean(np.abs(W), axis=-1)
        np.testing.assert_allclose(qw.scale, expected_scale, atol=1e-5)

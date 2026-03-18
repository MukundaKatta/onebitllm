"""Convert model checkpoints to quantized format with calibration data."""
import logging
import os
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .quantize import BitQuantizer, QuantizedWeight, dequantize

logger = logging.getLogger(__name__)


class ModelConverter:
    """Convert standard model weights to ultra-low-bit quantized format.

    Supports layer-by-layer quantization with optional calibration data
    for better accuracy.

    Args:
        bits: Target bit width.
        calibration_samples: Number of calibration samples to use.
        skip_patterns: Layer name patterns to keep in full precision.

    Example:
        converter = ModelConverter(bits=1.58)
        quantized = converter.convert(model_weights, calibration_data=X_cal)
        converter.save(quantized, "model_quantized/")
    """

    def __init__(
        self,
        bits: float = 1.58,
        calibration_samples: int = 128,
        skip_patterns: Optional[List[str]] = None,
        group_size: int = 128,
    ):
        self.bits = bits
        self.calibration_samples = calibration_samples
        self.skip_patterns = skip_patterns or ["embed", "layernorm", "ln_", "norm"]
        self.group_size = group_size
        self.quantizer = BitQuantizer(bits=bits, group_size=group_size)

    def convert(
        self,
        weights: Dict[str, np.ndarray],
        calibration_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Convert model weights to quantized format.

        Args:
            weights: Dict of layer names to float32 weight tensors.
            calibration_data: Optional calibration activations for
                range estimation.

        Returns:
            Dict with quantized layers and metadata.
        """
        quantized_layers: Dict[str, Any] = {}
        total_original = 0
        total_quantized = 0
        errors = []

        for name, weight in weights.items():
            original_bytes = weight.nbytes
            total_original += original_bytes

            # Skip certain layers
            if self._should_skip(name, weight):
                quantized_layers[name] = {
                    "type": "float32",
                    "data": weight,
                    "shape": weight.shape,
                }
                total_quantized += original_bytes
                continue

            # Apply calibration if available
            if calibration_data is not None:
                weight = self._calibrate_weight(weight, calibration_data, name)

            # Quantize
            qw = self.quantizer.quantize(weight)
            error = self.quantizer.compute_error(weight, qw)
            errors.append({"layer": name, **error})

            quantized_bytes = qw.data.nbytes + qw.scale.nbytes
            total_quantized += quantized_bytes

            quantized_layers[name] = {
                "type": f"quantized_{self.bits}bit",
                "data": qw.data,
                "scale": qw.scale,
                "zero_point": qw.zero_point,
                "bits": qw.bits,
                "method": qw.method,
                "shape": weight.shape,
            }

            logger.debug(f"Quantized '{name}': {weight.shape} -> SNR={error['snr_db']:.1f}dB")

        compression = total_original / max(total_quantized, 1)
        avg_snr = float(np.mean([e["snr_db"] for e in errors])) if errors else 0

        return {
            "layers": quantized_layers,
            "metadata": {
                "bits": self.bits,
                "n_layers": len(quantized_layers),
                "n_quantized": sum(1 for v in quantized_layers.values() if "quantized" in v["type"]),
                "n_float": sum(1 for v in quantized_layers.values() if v["type"] == "float32"),
                "original_mb": round(total_original / (1024 ** 2), 2),
                "quantized_mb": round(total_quantized / (1024 ** 2), 2),
                "compression_ratio": round(compression, 2),
                "avg_snr_db": round(avg_snr, 2),
            },
            "errors": errors,
        }

    def save(self, converted: Dict[str, Any], output_dir: str) -> str:
        """Save converted model to disk.

        Args:
            converted: Output from convert().
            output_dir: Directory to save to.

        Returns:
            Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save each layer
        arrays_to_save = {}
        layer_info = {}

        for name, layer in converted["layers"].items():
            safe_name = name.replace(".", "_")
            if layer["type"] == "float32":
                arrays_to_save[f"{safe_name}_data"] = layer["data"]
                layer_info[name] = {"type": "float32", "shape": list(layer["shape"]),
                                     "key": f"{safe_name}_data"}
            else:
                arrays_to_save[f"{safe_name}_data"] = layer["data"]
                arrays_to_save[f"{safe_name}_scale"] = layer["scale"]
                info = {
                    "type": layer["type"], "bits": layer["bits"], "method": layer["method"],
                    "shape": list(layer["shape"]), "key_data": f"{safe_name}_data",
                    "key_scale": f"{safe_name}_scale",
                }
                if layer.get("zero_point") is not None:
                    arrays_to_save[f"{safe_name}_zp"] = layer["zero_point"]
                    info["key_zp"] = f"{safe_name}_zp"
                layer_info[name] = info

        np.savez_compressed(os.path.join(output_dir, "weights.npz"), **arrays_to_save)

        manifest = {"metadata": converted["metadata"], "layers": layer_info}
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Saved quantized model to {output_dir}")
        return output_dir

    def load(self, model_dir: str) -> Dict[str, np.ndarray]:
        """Load a quantized model and dequantize for inference."""
        data = np.load(os.path.join(model_dir, "weights.npz"), allow_pickle=True)
        with open(os.path.join(model_dir, "manifest.json")) as f:
            manifest = json.load(f)

        weights = {}
        for name, info in manifest["layers"].items():
            if info["type"] == "float32":
                weights[name] = data[info["key"]]
            else:
                qw = QuantizedWeight(
                    data=data[info["key_data"]],
                    scale=data[info["key_scale"]],
                    zero_point=data.get(info.get("key_zp")),
                    bits=info["bits"],
                    original_shape=tuple(info["shape"]),
                    method=info["method"],
                )
                weights[name] = dequantize(qw)

        return weights

    def _should_skip(self, name: str, weight: np.ndarray) -> bool:
        if weight.ndim < 2:
            return True
        if any(p in name.lower() for p in self.skip_patterns):
            return True
        if weight.size < 256:
            return True
        return False

    def _calibrate_weight(
        self, weight: np.ndarray, calibration: np.ndarray, name: str
    ) -> np.ndarray:
        """Apply calibration-based scaling to improve quantization accuracy.

        Uses activation magnitudes to scale weights proportionally.
        """
        if calibration.shape[-1] == weight.shape[-1]:
            act_scale = np.std(calibration, axis=tuple(range(calibration.ndim - 1)))
            act_scale = np.clip(act_scale, 1e-6, None)
            # Smooth quantization: scale weights by activation magnitudes
            weight = weight * (act_scale / np.mean(act_scale))
        return weight

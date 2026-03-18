# onebitllm

**Inference framework optimized for 1-bit and ultra-low-bit quantized large language models**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Onebitllm
 instance = Onebitllm()
r = instance.quantize_layer(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `quantize_layer()` | Quantize layer |
| `pack_weights()` | Pack weights |
| `run_inference()` | Run inference |
| `benchmark_speed()` | Benchmark speed |
| `measure_perplexity()` | Measure perplexity |
| `convert_model()` | Convert model |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.

"""CLI for onebitllm."""
import sys, json, argparse
from .core import Onebitllm

def main():
    parser = argparse.ArgumentParser(description="Inference framework optimized for 1-bit and ultra-low-bit quantized large language models")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Onebitllm()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.quantize_layer(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"onebitllm v0.1.0 — Inference framework optimized for 1-bit and ultra-low-bit quantized large language models")

if __name__ == "__main__":
    main()

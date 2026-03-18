"""Basic usage example for onebitllm."""
from src.core import Onebitllm

def main():
    instance = Onebitllm(config={"verbose": True})

    print("=== onebitllm Example ===\n")

    # Run primary operation
    result = instance.quantize_layer(input="example data", mode="demo")
    print(f"Result: {result}")

    # Run multiple operations
    ops = ["quantize_layer", "pack_weights", "run_inference]
    for op in ops:
        r = getattr(instance, op)(source="example")
        print(f"  {op}: {"✓" if r.get("ok") else "✗"}")

    # Check stats
    print(f"\nStats: {instance.get_stats()}")

if __name__ == "__main__":
    main()

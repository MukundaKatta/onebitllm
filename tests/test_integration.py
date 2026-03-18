"""Integration tests for Onebitllm."""
from src.core import Onebitllm

class TestOnebitllm:
    def setup_method(self):
        self.c = Onebitllm()
    def test_10_ops(self):
        for i in range(10): self.c.quantize_layer(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.quantize_layer()["service"] == "onebitllm"
    def test_different_inputs(self):
        self.c.quantize_layer(type="a"); self.c.quantize_layer(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Onebitllm(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.quantize_layer()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.quantize_layer()
        assert self.c.get_stats()["ops"] == 100

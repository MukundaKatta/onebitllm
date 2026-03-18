from src.core import BitQuantizer
def test_init(): assert BitQuantizer().get_stats()["ops"] == 0
def test_op(): c = BitQuantizer(); c.quantize_layer(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = BitQuantizer(); [c.quantize_layer() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = BitQuantizer(); c.quantize_layer(); c.reset(); assert c.get_stats()["ops"] == 0

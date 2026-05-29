# tests/distill/test_adapters.py
import torch
from eupe.distill.adapters import AdapterHead, AdapterHeadSet

def test_adapter_head_structure_and_shapes():
    h = AdapterHead(16, 32, 24)
    assert h.fc1.bias is None and h.fc2.bias is None          # no-bias (paper §4.1)
    assert isinstance(h.norm, torch.nn.LayerNorm)
    assert h(torch.randn(4, 16)).shape == (4, 24)             # cls path
    assert h(torch.randn(4, 7, 16)).shape == (4, 7, 24)       # patch path

def test_adapter_head_set_routes_per_teacher():
    s = AdapterHeadSet(student_dim=16, teacher_specs=[("t_a", 24), ("t_b", 40)], hidden_dim=32)
    out = s(torch.randn(2, 16), torch.randn(2, 5, 16))
    assert set(out) == {"t_a", "t_b"}
    assert out["t_a"]["cls"].shape == (2, 24) and out["t_a"]["patch"].shape == (2, 5, 24)
    assert out["t_b"]["cls"].shape == (2, 40) and out["t_b"]["patch"].shape == (2, 5, 40)

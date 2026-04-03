import mlx.core as mx
import mlx.nn as nn
import pytest
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader, PatternMatch
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

def test_lokr_forward_full():
    # Test LokrLinear with full w1, w2 matrices
    input_dims = 16
    output_dims = 16 # Adjust to match kron product shape (2*8, 2*8) if we want a square delta
    
    # w1 (2, 2), w2 (8, 8) -> Result (16, 16)
    w1 = mx.random.uniform(shape=(2, 2))
    w2 = mx.random.uniform(shape=(8, 8))
    
    linear = nn.Linear(input_dims, output_dims)
    lokr = LokrLinear.from_linear(linear, scale=2.0)
    lokr.lokr_w1 = w1
    lokr.lokr_w2 = w2
    
    x = mx.random.uniform(shape=(1, input_dims))
    
    # Expected output calculation
    delta_w = mx.kron(w1, w2) # (2*8, 2*8) = (16, 16)
    expected_out = linear(x) + 2.0 * mx.matmul(x, delta_w.T)
    
    actual_out = lokr(x)
    
    assert mx.allclose(actual_out, expected_out, atol=1e-5)

def test_lokr_forward_factorized():
    # Test LokrLinear with factorized matrices (w1_a, w1_b)
    input_dims = 16
    output_dims = 16
    
    w1_a = mx.random.uniform(shape=(2, 2))
    w1_b = mx.random.uniform(shape=(2, 2))
    w2 = mx.random.uniform(shape=(8, 8))
    
    linear = nn.Linear(input_dims, output_dims)
    lokr = LokrLinear.from_linear(linear, scale=1.0)
    lokr.lokr_w1_a = w1_a
    lokr.lokr_w1_b = w1_b
    lokr.lokr_w2 = w2
    
    x = mx.random.uniform(shape=(1, input_dims))
    
    w1 = mx.matmul(w1_a, w1_b)
    delta_w = mx.kron(w1, w2)
    expected_out = linear(x) + mx.matmul(x, delta_w.T)
    
    actual_out = lokr(x)
    
    assert mx.allclose(actual_out, expected_out, atol=1e-5)

def test_lora_loader_lokr_pattern_generation():
    # Test that LoRALoader correctly derives Lokr patterns from LoRA targets
    target = LoRATarget(
        model_path="test.path",
        possible_up_patterns=["test.up"],
        possible_down_patterns=["test.down"]
    )
    
    mappings = LoRALoader._build_pattern_mappings([target])
    
    # LoRALoader appends suffixes to the base path derived from possible_down_patterns
    # "test.down" -> base "test.down" (since it doesn't end in .lora_A.weight etc) -> "test.down.lokr_w1"
    lokr_patterns = [m.source_pattern for m in mappings if "lokr_" in m.source_pattern]
    
    assert "test.down.lokr_w1" in lokr_patterns
    assert "test.down.lokr_w2" in lokr_patterns
    assert "test.down.lokr_w1_a" in lokr_patterns

def test_apply_lokr_to_target():
    # Test _apply_lora_matrices_to_target with Lokr data
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16, 16)
            
    model = MockModel()
    
    # Kronecker: (2, 2) x (8, 8) -> (16, 16)
    lora_data = {
        "lokr_w1": mx.random.uniform(shape=(2, 2)),
        "lokr_w2": mx.random.uniform(shape=(8, 8)),
    }
    
    success = LoRALoader._apply_lora_matrices_to_target(
        model, "layer", lora_data, scale=1.0, role="test"
    )
    
    assert success
    assert isinstance(model.layer, LokrLinear)
    assert model.layer._mflux_lora_role == "test"
    assert model.layer.lokr_w1 is not None
    assert model.layer.lokr_w2 is not None

def test_fused_mixed_adapters():
    # Test FusedLoRALinear with both LoRA and Lokr
    input_dims = 16
    output_dims = 16
    base_linear = nn.Linear(input_dims, output_dims)
    
    # LoRA component
    lora = LoRALinear.from_linear(base_linear, r=4, scale=1.0)
    lora.lora_A = mx.random.uniform(shape=(16, 4))
    lora.lora_B = mx.random.uniform(shape=(4, 16))
    
    # Lokr component
    lokr = LokrLinear.from_linear(base_linear, scale=1.0)
    lokr.lokr_w1 = mx.random.uniform(shape=(2, 2))
    lokr.lokr_w2 = mx.random.uniform(shape=(8, 8))
    
    fused = FusedLoRALinear(base_linear, [lora, lokr])
    
    x = mx.random.uniform(shape=(1, input_dims))
    
    # Manual calculation
    base_out = base_linear(x)
    lora_delta = mx.matmul(mx.matmul(x, lora.lora_A), lora.lora_B)
    lokr_delta = mx.matmul(x, mx.kron(lokr.lokr_w1, lokr.lokr_w2).T)
    expected_out = base_out + lora_delta + lokr_delta
    
    actual_out = fused(x)
    
    assert mx.allclose(actual_out, expected_out, atol=1e-5)

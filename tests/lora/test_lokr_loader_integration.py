import mlx.core as mx
from mlx import nn
import pytest
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

def test_lokr_loader_padding_integration():
    # 1. Base linear layer for to_qkv_mlp_proj (Out=36864, In=4096)
    base_linear = nn.Linear(4096, 36864, bias=False)
    
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 2. Mock LoRATarget with the padding transform
    from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms
    
    target = LoRATarget(
        model_path="layer",
        possible_up_patterns=["layer.lokr_w1"], # Using simplified patterns for test
        possible_down_patterns=["layer.lokr_w1"], # The loader derives lokr keys from down patterns
        possible_alpha_patterns=[],
        up_transform=LoraTransforms.pad_klein9b_single_linear1_up,
        down_transform=None
    )
    
    # Override patterns because the loader derives them
    # The loader expects the base pattern to match.
    target.possible_down_patterns = ["layer.lora_down.weight"] 
    
    # 3. Dummy LoKr weights (unpadded)
    # w1=(96, 32), w2=(128, 128) -> Out=12288, In=4096
    weights = {
        "layer.lokr_w1": mx.zeros((96, 32)),
        "layer.lokr_w2": mx.zeros((128, 128))
    }
    
    # 4. Apply using the loader's mapping logic
    # This will trigger _build_pattern_mappings and _apply_lora_with_mapping
    # The transform should be automatically applied now!
    pattern_mappings = LoRALoader._build_pattern_mappings([target])
    
    applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(
        transformer=transformer,
        weights=weights,
        scale=1.0,
        pattern_mappings=pattern_mappings,
        role=None
    )
    
    assert applied_count == 1
    assert "layer.lokr_w1" in matched_keys
    assert "layer.lokr_w2" in matched_keys
    
    # Verify the layer was replaced and has padded weights
    assert isinstance(transformer.layer, LokrLinear)
    assert transformer.layer.lokr_w1.shape == (32, 288) # In=32, Out=96+192=288
    
    # Test forward pass
    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 36864)

if __name__ == "__main__":
    pytest.main([__file__])

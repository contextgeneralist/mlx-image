import mlx.core as mx
from mlx import nn
import pytest
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear

def test_lora_shape_mismatch_validation():
    # 1. Create a base linear layer (representing a layer in a Flux 2 model with dim 4096)
    # Output dim 12288 (e.g. fused qkv), Input dim 4096
    base_linear = nn.Linear(4096, 12288, bias=False)
    
    # 2. Create a dummy transformer model containing this layer
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 3. Create dummy LoRA weights that mismatch (e.g. Flux 1 weights with dim 3072)
    # lora_A: (In, Rank) -> (3072, 16)
    # lora_B: (Rank, Out) -> (16, 9216)
    lora_data = {
        "lora_A": mx.zeros((3072, 16)),
        "lora_B": mx.zeros((16, 9216))
    }
    
    # 4. Attempt to apply these weights to the layer
    # This should fail validation in LoRALoader
    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None
    )
    
    # Currently, this is expected to return True but result in a broken layer
    # After the fix, it should return False
    assert success is False

def test_klein9b_single_block_to_out_padding():
    # 1. Base linear layer for to_out (Out=4096, In=16384)
    base_linear = nn.Linear(16384, 4096, bias=False)
    
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 2. LoKr weights that match the UN-FUSED attention output part
    # w1=(Out1, In1)=(32, 32), w2=(128, 128) -> Out=4096, In=4096
    lora_data = {
        "lokr_w1": mx.zeros((32, 32)),
        "lokr_w2": mx.zeros((128, 128))
    }
    
    # 3. Apply the transform manually (In=4096 -> 16384)
    from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms
    
    transformed_lora_data = {
        "lokr_w1": LoraTransforms.pad_klein9b_single_linear2_down(lora_data["lokr_w1"]),
        "lokr_w2": lora_data["lokr_w2"]
    }
    
    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=transformed_lora_data,
        scale=1.0,
        role=None
    )
    
    # This should now succeed!
    assert success is True
    assert isinstance(transformer.layer, LokrLinear)
    
    # Test forward pass
    x = mx.zeros((1, 4608, 16384))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 4096)

def test_klein9b_single_block_padding():
    # 1. Base linear layer for to_qkv_mlp_proj (Out=36864, In=4096)
    base_linear = nn.Linear(4096, 36864, bias=False)
    
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 2. LoKr weights that match the UN-FUSED QKV part
    # w1=(Out1, In1)=(96, 32), w2=(128, 128) -> Out=12288, In=4096
    lora_data = {
        "lokr_w1": mx.zeros((96, 32)),
        "lokr_w2": mx.zeros((128, 128))
    }
    
    # 3. Apply the transform manually as the loader would do during pattern matching
    from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms
    
    transformed_lora_data = {
        "lokr_w1": LoraTransforms.pad_klein9b_single_linear1_up(lora_data["lokr_w1"]),
        "lokr_w2": lora_data["lokr_w2"]
    }
    
    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=transformed_lora_data,
        scale=1.0,
        role=None
    )
    
    # This should now succeed!
    assert success is True
    assert isinstance(transformer.layer, LokrLinear)
    
    # Test forward pass
    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 36864)

def test_lokr_shape_robustness():
    # 1. Base linear layer (Out=12288, In=4096)
    base_linear = nn.Linear(4096, 12288, bias=False)
    
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 2. Dummy LoKr weights that are swapped (In, Out) instead of (Out, In)
    # w1=(24, 72), w2=(128, 128)? 24*128=3072, 72*128=9216. (for Flux 1)
    # For Flux 2 Klein: Out=12288, In=4096. 
    # w1=(32, 96), w2=(128, 128)? 32*128=4096, 96*128=12288.
    lora_data = {
        "lokr_w1": mx.zeros((32, 96)),
        "lokr_w2": mx.zeros((128, 128))
    }
    
    # 3. Attempt to apply
    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None
    )
    
    # This should now succeed because LokrLinear handles swapped shapes!
    assert success is True
    
    # Verify the layer was replaced with a LokrLinear
    assert isinstance(transformer.layer, LokrLinear)
    
    # Test a forward pass doesn't crash
    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 12288)

def test_lokr_shape_mismatch_validation():
    # 1. Base linear layer (Out=12288, In=4096)
    base_linear = nn.Linear(4096, 12288, bias=False)
    
    class DummyTransformer(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.layer = linear
    
    transformer = DummyTransformer(base_linear)
    
    # 2. Dummy LoKr weights that mismatch
    # w1: (Out1, In1), w2: (Out2, In2)
    # In Flux 1: Out=9216, In=3072. 
    # Maybe w1=(72, 24), w2=(128, 128)? 72*128=9216, 24*128=3072.
    lora_data = {
        "lokr_w1": mx.zeros((72, 24)),
        "lokr_w2": mx.zeros((128, 128))
    }
    
    # 3. Attempt to apply
    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None
    )
    
    assert success is False

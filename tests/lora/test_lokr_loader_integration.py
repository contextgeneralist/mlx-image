import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms


def test_lokr_loader_padding_integration(DummyTransformer):
    """The loader applies up_transform to the kron-product delta before storing it.

    Scenario: A 3B Flux2 single-block adapter (kron result: 27648×4096) is loaded
    into a 9B model layer (36864×4096).  pad_flux2_single_linear1_up pads axis 0
    from 27648 → 36864.
    """
    base_linear = nn.Linear(4096, 36864, bias=False)
    transformer = DummyTransformer(base_linear)

    target = LoRATarget(
        model_path="layer",
        possible_up_patterns=["layer.lora_up.weight"],
        possible_down_patterns=["layer.lora_down.weight"],
        possible_alpha_patterns=[],
        up_transform=LoraTransforms.pad_flux2_single_linear1_up,
        down_transform=None,
    )

    # 3B LoKr factors: kron((216,32),(128,128)) = (27648, 4096)
    weights = {
        "layer.lokr_w1": mx.zeros((216, 32)),
        "layer.lokr_w2": mx.zeros((128, 128)),
    }

    pattern_mappings = LoRALoader._build_pattern_mappings([target])
    applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(
        transformer=transformer,
        weights=weights,
        scale=1.0,
        pattern_mappings=pattern_mappings,
        role=None,
    )

    assert applied_count == 1
    assert "layer.lokr_w1" in matched_keys
    assert "layer.lokr_w2" in matched_keys

    assert isinstance(transformer.layer, LokrLinear)
    # After padding, delta_w should match the 9B layer shape
    assert transformer.layer.delta_w.shape == (4096, 36864)

    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 36864)


if __name__ == "__main__":
    pytest.main([__file__])

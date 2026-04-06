import mlx.core as mx
from mlx import nn
import pytest
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear


def test_lora_shape_mismatch_validation(DummyTransformer):
    # Flux 2 layer: Out=12288 (fused QKV), In=4096
    base_linear = nn.Linear(4096, 12288, bias=False)
    transformer = DummyTransformer(base_linear)

    # Flux 1 weights with different dimensions — should be rejected
    lora_data = {
        "lora_A": mx.zeros((3072, 16)),
        "lora_B": mx.zeros((16, 9216)),
    }

    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None,
    )

    assert success is False


def test_flux2_single_block_to_out_padding(DummyTransformer):
    # linear2 (to_out) layer: Out=4096, In=16384
    # A 9B to_out adapter with w1=(32, 32) is padded to (128, 32) by
    # pad_flux2_single_linear2_down, giving kron size (16384, 4096).
    # The layer weight shape is (4096, 16384) so the transposed kron matches.
    base_linear = nn.Linear(16384, 4096, bias=False)
    transformer = DummyTransformer(base_linear)

    from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms

    lora_data = {
        "lokr_w1": LoraTransforms.pad_flux2_single_linear2_down(mx.zeros((32, 32))),
        "lokr_w2": mx.zeros((128, 128)),
    }

    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None,
    )

    assert success is True
    assert isinstance(transformer.layer, LokrLinear)

    x = mx.zeros((1, 4608, 16384))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 4096)


def test_flux2_single_block_padding(DummyTransformer):
    # linear1 (QKV+MLP) layer: Out=36864 (9B), In=4096
    # A 3B adapter produces kron((216, 32), (128, 128)) = (27648, 4096).
    # The up_transform pads axis 0: 27648 → 36864.
    base_linear = nn.Linear(4096, 36864, bias=False)
    transformer = DummyTransformer(base_linear)

    from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms

    # Pass raw 3B factors; the loader applies up_transform to the kron result.
    # kron((216, 32), (128, 128)) = (27648, 4096)
    # pad_flux2_single_linear1_up: 27648 → 36864
    lora_data = {
        "lokr_w1": mx.zeros((216, 32)),
        "lokr_w2": mx.zeros((128, 128)),
        "up_transform": LoraTransforms.pad_flux2_single_linear1_up,
        "transpose": True,
    }

    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None,
    )

    assert success is True
    assert isinstance(transformer.layer, LokrLinear)
    assert transformer.layer.delta_w.shape == (36864, 4096)

    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 36864)


def test_lokr_shape_robustness(DummyTransformer):
    # Layer: Out=12288, In=4096
    base_linear = nn.Linear(4096, 12288, bias=False)
    transformer = DummyTransformer(base_linear)

    # Factors saved as (In, Out) — swapped relative to convention.
    # kron((32, 96), (128, 128)) = (4096, 12288) = layer_shape[::-1]
    lora_data = {
        "lokr_w1": mx.zeros((32, 96)),
        "lokr_w2": mx.zeros((128, 128)),
    }

    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None,
    )

    assert success is True
    assert isinstance(transformer.layer, LokrLinear)

    x = mx.zeros((1, 4608, 4096))
    out = transformer.layer(x)
    assert out.shape == (1, 4608, 12288)


def test_lokr_shape_mismatch_validation(DummyTransformer):
    # Layer: Out=12288, In=4096
    base_linear = nn.Linear(4096, 12288, bias=False)
    transformer = DummyTransformer(base_linear)

    # Flux 1 dimensions — neither (12288, 4096) nor (4096, 12288)
    # kron((72, 24), (128, 128)) = (9216, 3072)
    lora_data = {
        "lokr_w1": mx.zeros((72, 24)),
        "lokr_w2": mx.zeros((128, 128)),
    }

    success = LoRALoader._apply_lora_matrices_to_target(
        transformer=transformer,
        target_path="layer",
        lora_data=lora_data,
        scale=1.0,
        role=None,
    )

    assert success is False

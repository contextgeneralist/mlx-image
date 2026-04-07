import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class FusedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list[LoRALinear | LokrLinear]):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras

    def __call__(self, x):
        base_out = self.base_linear(x)

        total_delta = mx.zeros_like(base_out)
        for adapter in self.loras:
            total_delta += adapter.compute_delta(x)

        return base_out + total_delta

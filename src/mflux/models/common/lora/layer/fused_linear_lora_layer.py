import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear


class FusedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list[LoRALinear | LokrLinear]):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras

    def __call__(self, x):
        base_out = self.base_linear(x)

        total_delta = mx.zeros_like(base_out)
        for adapter in self.loras:
            if isinstance(adapter, LoRALinear):
                total_delta += adapter.scale * mx.matmul(mx.matmul(x, adapter.lora_A), adapter.lora_B)
            elif isinstance(adapter, LokrLinear):
                # Delegate to the adapter's __call__ so delta_w and factor-based
                # paths are handled consistently without duplicating logic here.
                total_delta += adapter(x) - adapter.linear(x)

        return base_out + total_delta

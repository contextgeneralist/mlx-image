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
                # 1. Compute W1
                if adapter.lokr_w1 is not None:
                    w1 = adapter.lokr_w1
                elif adapter.lokr_w1_a is not None and adapter.lokr_w1_b is not None:
                    w1 = mx.matmul(adapter.lokr_w1_a, adapter.lokr_w1_b)
                else:
                    continue

                # 2. Compute W2
                if adapter.lokr_w2 is not None:
                    w2 = adapter.lokr_w2
                elif adapter.lokr_w2_a is not None and adapter.lokr_w2_b is not None:
                    w2 = mx.matmul(adapter.lokr_w2_a, adapter.lokr_w2_b)
                else:
                    continue

                # 3. Apply Kronecker Product delta
                delta_w = mx.kron(w1, w2)
                total_delta += adapter.scale * mx.matmul(x, delta_w.T)

        return base_out + total_delta

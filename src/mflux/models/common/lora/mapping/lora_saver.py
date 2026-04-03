import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear


class LoRASaver:
    @staticmethod
    def bake_and_strip_lora(module: nn.Module) -> nn.Module:
        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _bake_single(adapter_layer: LoRALinear | LokrLinear) -> nn.Module:
            base_linear = adapter_layer.linear
            if isinstance(adapter_layer, LoRALinear):
                LoRASaver._apply_lora_delta(base_linear, adapter_layer)
            elif isinstance(adapter_layer, LokrLinear):
                LoRASaver._apply_lokr_delta(base_linear, adapter_layer)
            return base_linear

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            base_linear = fused_layer.base_linear
            for adapter in fused_layer.loras:
                if isinstance(adapter, LoRALinear):
                    LoRASaver._apply_lora_delta(base_linear, adapter)
                elif isinstance(adapter, LokrLinear):
                    LoRASaver._apply_lokr_delta(base_linear, adapter)
            return base_linear

        def _walk(obj, parent=None, attr_name=None, idx=None):
            # Replace wrappers first
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, (LoRALinear, LokrLinear)):
                new_child = _bake_single(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child

            # Recurse into containers/modules
            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i)
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i)
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None)
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None)

        _walk(module, None, None, None)
        return module

    @staticmethod
    def _apply_lora_delta(base_linear: nn.Module, lora_layer: LoRALinear) -> None:
        if not hasattr(base_linear, "weight"):
            return

        weight = base_linear.weight
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)  # shape: [in, out]
        delta = mx.transpose(delta)  # shape: [out, in]
        delta = lora_layer.scale * delta

        if weight.shape != delta.shape:
            print(f"⚠️  Skipping LoRA bake due to shape mismatch: weight {weight.shape} vs delta {delta.shape}")
            return

        try:
            base_linear.weight = weight + delta.astype(weight.dtype)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake LoRA into base layer: {e}")

    @staticmethod
    def _apply_lokr_delta(base_linear: nn.Module, lokr_layer: LokrLinear) -> None:
        if not hasattr(base_linear, "weight"):
            return

        # 1. Compute W1
        if lokr_layer.lokr_w1 is not None:
            w1 = lokr_layer.lokr_w1
        elif lokr_layer.lokr_w1_a is not None and lokr_layer.lokr_w1_b is not None:
            w1 = mx.matmul(lokr_layer.lokr_w1_a, lokr_layer.lokr_w1_b)
        else:
            return

        # 2. Compute W2
        if lokr_layer.lokr_w2 is not None:
            w2 = lokr_layer.lokr_w2
        elif lokr_layer.lokr_w2_a is not None and lokr_layer.lokr_w2_b is not None:
            w2 = mx.matmul(lokr_layer.lokr_w2_a, lokr_layer.lokr_w2_b)
        else:
            return

        weight = base_linear.weight
        delta = mx.kron(w1, w2)
        delta = lokr_layer.scale * delta

        if weight.shape != delta.shape:
            # Try transpose
            if weight.shape == delta.T.shape:
                delta = delta.T
            else:
                print(f"⚠️  Skipping Lokr bake due to shape mismatch: weight {weight.shape} vs delta {delta.shape}")
                return

        try:
            base_linear.weight = weight + delta.astype(weight.dtype)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake Lokr into base layer: {e}")

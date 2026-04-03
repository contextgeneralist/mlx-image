from __future__ import annotations

import math

from collections.abc import Iterator
from contextlib import contextmanager

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.training.lora.path_util import get_at_path


class TrainingUtil:
    @staticmethod
    def iter_assistant_loras(transformer) -> Iterator[LoRALinear | LokrLinear]:
        for _, child in transformer.named_modules():
            if isinstance(child, (LoRALinear, LokrLinear)):
                if getattr(child, "_mflux_lora_role", None) == "assistant":
                    yield child
            elif isinstance(child, FusedLoRALinear):
                for adapter in child.loras:
                    if getattr(adapter, "_mflux_lora_role", None) == "assistant":
                        yield adapter

    @staticmethod
    @contextmanager
    def assistant_disabled(transformer):
        saved_scales = [(adapter, float(adapter.scale)) for adapter in TrainingUtil.iter_assistant_loras(transformer)]
        for adapter, _ in saved_scales:
            adapter.scale = 0.0
        try:
            yield
        finally:
            for mod, s in saved_scales:
                mod.scale = s

    @staticmethod
    def get_train_lora(transformer, module_path: str) -> LoRALinear | LokrLinear:
        current = get_at_path(transformer, module_path)
        if isinstance(current, (LoRALinear, LokrLinear)):
            if getattr(current, "_mflux_lora_role", None) == "train":
                return current
        elif isinstance(current, FusedLoRALinear):
            for adapter in current.loras:
                if getattr(adapter, "_mflux_lora_role", None) == "train":
                    return adapter

        raise ValueError(
            f"Expected a trainable adapter at '{module_path}' but found {type(current)} (or no train adapter in fusion)."
        )

    @staticmethod
    def resolve_dimensions(
        *,
        width: int,
        height: int,
        max_resolution: int | None,
        default_max_resolution: int | None = None,
        error_template: str | None = None,
    ) -> tuple[int, int]:
        effective_max = max_resolution if max_resolution is not None else default_max_resolution
        if effective_max is not None:
            max_area = effective_max * effective_max
            current_area = width * height
            if current_area > max_area:
                scale = math.sqrt(max_area / current_area)
                width = int(width * scale)
                height = int(height * scale)

        adj_width = 16 * (int(width) // 16)
        adj_height = 16 * (int(height) // 16)
        if adj_width <= 0 or adj_height <= 0:
            if error_template:
                raise ValueError(error_template.format(width=width, height=height))
            raise ValueError("Image too small for training (needs >=16px).")

        return adj_width, adj_height

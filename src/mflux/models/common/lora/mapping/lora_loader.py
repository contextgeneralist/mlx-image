import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LokrLinear
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.resolution.lora_resolution import LoraResolution

# All weight keys that can appear in a LoKr adapter file
_LOKR_WEIGHT_KEYS = ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2", "dora_scale")


def _quantized_input_dims(linear: nn.Linear | nn.QuantizedLinear) -> int:
    """Return the logical input dimension of a linear layer.

    QuantizedLinear packs values at `bits`-per-element into 32-bit words,
    so the stored weight column count must be expanded by 32 // bits.
    """
    dims = linear.weight.shape[1]
    if isinstance(linear, nn.QuantizedLinear):
        dims *= 32 // linear.bits
    return dims


@dataclass
class PatternMatch:
    source_pattern: str
    target_path: str
    matrix_name: str  # "lora_A", "lora_B", or "alpha"
    transpose: bool
    transform: Callable[[mx.array], mx.array] | None = None
    up_transform: Callable[[mx.array], mx.array] | None = None
    down_transform: Callable[[mx.array], mx.array] | None = None


class LoRALoader:
    @staticmethod
    def load_and_apply_lora(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        role: str | None = None,
    ) -> tuple[list[str], list[float]]:
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        if not resolved_paths:
            return resolved_paths, []

        resolved_scales = LoraResolution.resolve_scales(lora_scales, len(resolved_paths))
        if len(resolved_scales) != len(resolved_paths):
            raise ValueError(
                f"Number of LoRA scales ({len(resolved_scales)}) must match number of LoRA files ({len(resolved_paths)})"
            )

        print(f"📦 Loading {len(resolved_paths)} LoRA file(s)...")

        for lora_file, scale in zip(resolved_paths, resolved_scales):
            LoRALoader._apply_single_lora(transformer, lora_file, scale, lora_mapping, role=role)

        print("✅ All LoRA weights applied successfully")

        return resolved_paths, resolved_scales

    @staticmethod
    def _apply_single_lora(
        transformer: nn.Module,
        lora_file: str,
        scale: float,
        lora_mapping: list[LoRATarget],
        *,
        role: str | None,
    ) -> None:
        # Load the LoRA weights
        if not Path(lora_file).exists():
            print(f"❌ LoRA file not found: {lora_file}")
            return

        print(f"🔧 Applying LoRA: {Path(lora_file).name} (scale={scale})")

        try:
            weights = dict(mx.load(lora_file, return_metadata=True)[0].items())
            
            # Normalize third-party keys (AIToolkit, Diffusers, XLabs) to standard format
            from mflux.models.common.lora.mapping.lora_normalizer import LoRANormalizer
            weights = LoRANormalizer.normalize(weights)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to load LoRA file: {e}")
            return

        # Build pattern mappings from LoRATargets
        pattern_mappings = LoRALoader._build_pattern_mappings(lora_mapping)

        # Apply LoRA using the mappings (allows multiple targets per source)
        applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(
            transformer, weights, scale, pattern_mappings, role=role
        )

        # Report results
        total_keys = len(weights)
        unmatched_keys = set(weights.keys()) - matched_keys

        print(f"   ✅ Applied to {applied_count} layers ({len(matched_keys)}/{total_keys} keys matched)")

        if unmatched_keys:
            print(f"   ⚠️  {len(unmatched_keys)} unmatched keys in LoRA file:")
            for key in sorted(unmatched_keys)[:5]:
                print(f"      - {key}")
            if len(unmatched_keys) > 5:
                print(f"      ... and {len(unmatched_keys) - 5} more")

    @staticmethod
    def _build_pattern_mappings(targets: list[LoRATarget]) -> list[PatternMatch]:
        mappings = []

        for target in targets:
            # Add up weight patterns (lora_B)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_B",
                    transpose=True,
                    transform=target.up_transform,
                )
                for pattern in target.possible_up_patterns
            )

            # Add down weight patterns (lora_A)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_A",
                    transpose=True,
                    transform=target.down_transform,
                )
                for pattern in target.possible_down_patterns
            )

            # Add alpha patterns (no transpose, no transform)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="alpha",
                    transpose=False,
                    transform=None,
                )
                for pattern in target.possible_alpha_patterns
            )

            # Derive LoKr patterns from LoRA down patterns by stripping LoRA suffixes
            for pattern in target.possible_down_patterns:
                base = pattern.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
                base = base.replace(".lora_A", "").replace(".lora_down", "")

                for key in _LOKR_WEIGHT_KEYS:
                    for suffix in ["", ".weight"]:
                        mappings.append(PatternMatch(
                            source_pattern=f"{base}.{key}{suffix}",
                            target_path=target.model_path,
                            matrix_name=key,
                            transpose=True,
                            transform=None,
                            up_transform=target.up_transform,
                            down_transform=target.down_transform,
                        ))

        return mappings

    @staticmethod
    def _apply_lora_with_mapping(
        transformer: nn.Module,
        weights: dict,
        scale: float,
        pattern_mappings: list[PatternMatch],
        *,
        role: str | None,
    ) -> tuple[int, set]:
        applied_count = 0
        lora_data_by_target: dict[str, dict] = {}
        matched_keys: set[str] = set()

        # For each weight key, find ALL matching patterns (not just first)
        # This allows multiple targets to use the same source (e.g., QKV split)
        for weight_key, weight_value in weights.items():
            for mapping in pattern_mappings:
                match_result = LoRALoader._match_pattern(weight_key, mapping.source_pattern)
                if match_result is None:
                    continue

                matched_keys.add(weight_key)
                block_idx = match_result

                # Resolve target path with block index if needed
                target_path = mapping.target_path
                if block_idx is not None and "{block}" in target_path:
                    target_path = target_path.format(block=block_idx)

                # Store for this target
                if target_path not in lora_data_by_target:
                    lora_data_by_target[target_path] = {}

                # Handle transforms
                if mapping.matrix_name.startswith("lokr_"):
                    # For LoKr, we store the transforms separately to apply them to the reconstructed matrix
                    if "up_transform" not in lora_data_by_target[target_path]:
                        lora_data_by_target[target_path]["up_transform"] = mapping.up_transform
                    if "down_transform" not in lora_data_by_target[target_path]:
                        lora_data_by_target[target_path]["down_transform"] = mapping.down_transform
                    
                    # For LoKr, we also need to know if it should be transposed eventually
                    lora_data_by_target[target_path]["transpose"] = mapping.transpose
                    
                    # Store raw value (do NOT transpose factors yet)
                    lora_data_by_target[target_path][mapping.matrix_name] = weight_value
                else:
                    # Regular LoRA
                    transformed_value = weight_value
                    if mapping.transform is not None:
                        transformed_value = mapping.transform(weight_value)
                    if mapping.transpose:
                        transformed_value = transformed_value.T
                    lora_data_by_target[target_path][mapping.matrix_name] = transformed_value

        # Apply LoRA to each target
        for target_path, lora_data in lora_data_by_target.items():
            if LoRALoader._apply_lora_matrices_to_target(transformer, target_path, lora_data, scale, role=role):
                applied_count += 1

        return applied_count, matched_keys

    @staticmethod
    def _match_pattern(weight_key: str, pattern: str) -> int | None:
        if "{block}" in pattern:
            # Find all numbers in the weight key
            numbers_in_key = re.findall(r"\d+", weight_key)
            for num_str in numbers_in_key:
                test_block_idx = int(num_str)
                concrete_pattern = pattern.replace("{block}", str(test_block_idx))
                if weight_key == concrete_pattern:
                    return test_block_idx
            return None
        else:
            if weight_key == pattern:
                return 0  # Return 0 to indicate match (no block)
            return None

    @staticmethod
    def _extract_alpha(alpha_tensor: mx.array) -> float:
        val = alpha_tensor.item()
        # Some LoRA trainers incorrectly save float16 alpha values while the rest
        # of the file is saved as bfloat16, causing the safetensors header to label
        # the alpha tensor as bfloat16. A float16 value of 32.0 (0x5000) interpreted
        # as bfloat16 becomes ~8.58 billion. We fix this by bit-casting back.
        if val > 1e6 and alpha_tensor.dtype == mx.bfloat16:
            bytes_val = mx.view(alpha_tensor, mx.uint16)
            fp16_val = mx.view(bytes_val, mx.float16)
            return fp16_val.item()
        return float(val)

    @staticmethod
    def _apply_lora_matrices_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        # Navigate to the target layer
        current_module = transformer
        path_parts = target_path.split(".")

        try:
            for part in path_parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                elif isinstance(current_module, dict) and part in current_module:
                    current_module = current_module[part]
                else:
                    current_module = getattr(current_module, part)
        except (AttributeError, IndexError, KeyError):
            print(f"❌ Could not find target path: {target_path}")
            return False

        # Check if we have the required matrices
        is_lora_data = "lora_A" in lora_data and "lora_B" in lora_data
        is_lokr_data = any(k.startswith("lokr_") for k in lora_data.keys())

        if not is_lora_data and not is_lokr_data:
            print(f"❌ Missing required adapter matrices for {target_path}")
            return False

        # Calculate final scale - only use user scale, matching Diffusers approach
        effective_scale = scale

        # Check if it's a linear layer (either nn.Linear, LoRALinear, LokrLinear or FusedLoRALinear)
        is_linear = hasattr(current_module, "weight")
        is_lora_linear = isinstance(current_module, LoRALinear)
        is_lokr_linear = isinstance(current_module, LokrLinear)
        is_fused_linear = isinstance(current_module, FusedLoRALinear)

        if not (is_linear or is_lora_linear or is_lokr_linear or is_fused_linear):
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

        # Extract base linear
        if is_fused_linear:
            base_linear = current_module.base_linear
            existing_loras = current_module.loras
        elif is_lora_linear or is_lokr_linear:
            base_linear = current_module.linear
            existing_loras = [current_module]
        else:
            base_linear = current_module
            existing_loras = []

        # Validate shapes before applying
        if is_lora_data:
            lora_A = lora_data["lora_A"]
            lora_B = lora_data["lora_B"]
            
            # Check input dimension (lora_A.shape[0]) matches base linear input dimension
            layer_in_dims = _quantized_input_dims(base_linear)
            if lora_A.shape[0] != layer_in_dims:
                print(f"   ❌ Shape mismatch for {target_path}: LoRA input dims ({lora_A.shape[0]}) do not match layer input dims ({layer_in_dims})")
                return False
            
            # Check output dimension (lora_B.shape[1]) matches base linear output dimension
            layer_out_dims = base_linear.weight.shape[0]
            if lora_B.shape[1] != layer_out_dims:
                print(f"   ❌ Shape mismatch for {target_path}: LoRA output dims ({lora_B.shape[1]}) do not match layer output dims ({layer_out_dims})")
                return False
        
        if is_lokr_data:
            # Reconstruct full weight matrix via Kronecker product
            # 1. Reconstruct W2
            rank = None
            if "lokr_w2" in lora_data:
                w2 = lora_data["lokr_w2"]
            elif "lokr_w2_a" in lora_data and "lokr_w2_b" in lora_data:
                if "lokr_t2" in lora_data:
                    # Tucker decomposition: einsum("ijkl, jr, ip -> prkl", t2, w2_b, w2_a)
                    w2 = mx.einsum("ijkl, jr, ip -> prkl", lora_data["lokr_t2"], lora_data["lokr_w2_b"], lora_data["lokr_w2_a"])
                else:
                    w2 = mx.matmul(lora_data["lokr_w2_a"], lora_data["lokr_w2_b"])
                rank = lora_data["lokr_w2_b"].shape[0]
            else:
                w2 = None

            # 2. Reconstruct W1
            if "lokr_w1" in lora_data:
                w1 = lora_data["lokr_w1"]
            elif "lokr_w1_a" in lora_data and "lokr_w1_b" in lora_data:
                w1 = mx.matmul(lora_data["lokr_w1_a"], lora_data["lokr_w1_b"])
                if rank is None:
                    rank = lora_data["lokr_w1_b"].shape[0]
            else:
                w1 = None

            if w1 is None or w2 is None:
                print(f"   ❌ Missing LoKr factors for {target_path}: need lokr_w1 (or w1_a/w1_b) and lokr_w2 (or w2_a/w2_b)")
                return False

            # 3. Align dimensions for Conv layers
            if len(w2.shape) == 4:
                w1 = mx.expand_dims(mx.expand_dims(w1, 2), 3)

            # Reconstruct the full weight delta via Kronecker product
            delta_w = mx.kron(w1, w2)

            layer_shape = (base_linear.weight.shape[0], _quantized_input_dims(base_linear))

            # LoKr factors are generally stored such that kron(w1, w2) matches the weight matrix.
            # We want delta_w to be (Out, In) before applying transforms (padding/splitting).
            if delta_w.shape == layer_shape[::-1] and delta_w.shape != layer_shape:
                delta_w = delta_w.T

            # Apply transforms to the full reconstructed matrix.
            # up_transform pads/slices the output dimension (axis 0).
            # down_transform pads/slices the input dimension (axis 1).
            if lora_data.get("up_transform"):
                delta_w = lora_data["up_transform"](delta_w)

            if lora_data.get("down_transform"):
                delta_w = lora_data["down_transform"](delta_w)

            # Validate shape. It should now match either (Out, In) or (In, Out).
            delta_w_shape = delta_w.shape
            if delta_w_shape != layer_shape and delta_w_shape != layer_shape[::-1]:
                print(f"   ❌ Shape mismatch for {target_path}: LoKr shape {delta_w_shape} does not match layer shape {layer_shape}")
                return False

            # Ensure final delta_w is in the expected orientation (In, Out) for LokrLinear's matmul(x, delta_w).
            # If it's currently (Out, In), transpose it.
            if delta_w.shape == layer_shape:
                delta_w = delta_w.T

            # Apply alpha scaling: scale = alpha / rank
            if "alpha" in lora_data and rank is not None:
                alpha_value = LoRALoader._extract_alpha(lora_data["alpha"])
                delta_w = delta_w * (alpha_value / rank)

            if "dora_scale" in lora_data:
                print(f"   ⚠️  Warning: DoRA-LoKr is not fully supported for {target_path}; dora_scale is ignored.")

            lora_data["delta_w"] = delta_w

        # Create new adapter layer
        if is_lora_data:
            lora_A = lora_data["lora_A"]
            lora_B = lora_data["lora_B"]
            
            # Handle alpha scaling
            alpha_scale = 1.0
            if "alpha" in lora_data:
                alpha_value = LoRALoader._extract_alpha(lora_data["alpha"])
                rank = lora_A.shape[1]
                alpha_scale = alpha_value / rank

            adapter_layer = LoRALinear.from_linear(base_linear, r=lora_A.shape[1], scale=effective_scale)
            adapter_layer.lora_A = lora_A
            adapter_layer.lora_B = lora_B
            if "alpha" in lora_data:
                adapter_layer.lora_B = adapter_layer.lora_B * alpha_scale
        else:
            adapter_layer = LokrLinear.from_linear(base_linear, scale=effective_scale)
            adapter_layer.delta_w = lora_data["delta_w"]

        adapter_layer._mflux_lora_role = role

        # Wrap in Fusion if needed
        if existing_loras:
            print(f"   🔀 Fusing with existing adapters at {target_path}")
            replacement_layer = FusedLoRALinear(base_linear=base_linear, loras=existing_loras + [adapter_layer])
        else:
            replacement_layer = adapter_layer

        # Replace the layer in the parent module
        parent_module = transformer
        for part in path_parts[:-1]:
            if part.isdigit():
                parent_module = parent_module[int(part)]
            elif isinstance(parent_module, dict) and part in parent_module:
                parent_module = parent_module[part]
            else:
                parent_module = getattr(parent_module, part)

        final_attr = path_parts[-1]
        if final_attr.isdigit():
            parent_module[int(final_attr)] = replacement_layer
        elif isinstance(parent_module, dict) and final_attr in parent_module:
            parent_module[final_attr] = replacement_layer
        else:
            setattr(parent_module, final_attr, replacement_layer)

        return True

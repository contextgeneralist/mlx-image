from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget
from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms


class Flux2LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []

        # Global layers
        targets.extend([
            LoRATarget(
                model_path="x_embedder",
                possible_up_patterns=["x_embedder.lora_B.weight", "x_embedder.lora_up.weight", "img_in.lora_B.weight", "img_in.lora_up.weight"],
                possible_down_patterns=["x_embedder.lora_A.weight", "x_embedder.lora_down.weight", "img_in.lora_A.weight", "img_in.lora_down.weight"],
                possible_alpha_patterns=["x_embedder.alpha", "img_in.alpha"],
            ),
            LoRATarget(
                model_path="context_embedder",
                possible_up_patterns=["context_embedder.lora_B.weight", "context_embedder.lora_up.weight", "txt_in.lora_B.weight", "txt_in.lora_up.weight"],
                possible_down_patterns=["context_embedder.lora_A.weight", "context_embedder.lora_down.weight", "txt_in.lora_A.weight", "txt_in.lora_down.weight"],
                possible_alpha_patterns=["context_embedder.alpha", "txt_in.alpha"],
            ),
            LoRATarget(
                model_path="time_guidance_embed.linear_1",
                possible_up_patterns=["time_guidance_embed.linear_1.lora_B.weight", "time_guidance_embed.linear_1.lora_up.weight", "time_in.in_layer.lora_B.weight", "time_in.in_layer.lora_up.weight"],
                possible_down_patterns=["time_guidance_embed.linear_1.lora_A.weight", "time_guidance_embed.linear_1.lora_down.weight", "time_in.in_layer.lora_A.weight", "time_in.in_layer.lora_down.weight"],
                possible_alpha_patterns=["time_guidance_embed.linear_1.alpha", "time_in.in_layer.alpha"],
            ),
            LoRATarget(
                model_path="time_guidance_embed.linear_2",
                possible_up_patterns=["time_guidance_embed.linear_2.lora_B.weight", "time_guidance_embed.linear_2.lora_up.weight", "time_in.out_layer.lora_B.weight", "time_in.out_layer.lora_up.weight"],
                possible_down_patterns=["time_guidance_embed.linear_2.lora_A.weight", "time_guidance_embed.linear_2.lora_down.weight", "time_in.out_layer.lora_A.weight", "time_in.out_layer.lora_down.weight"],
                possible_alpha_patterns=["time_guidance_embed.linear_2.alpha", "time_in.out_layer.alpha"],
            ),
            LoRATarget(
                model_path="proj_out",
                possible_up_patterns=["proj_out.lora_B.weight", "proj_out.lora_up.weight", "final_layer.linear.lora_B.weight", "final_layer.linear.lora_up.weight"],
                possible_down_patterns=["proj_out.lora_A.weight", "proj_out.lora_down.weight", "final_layer.linear.lora_A.weight", "final_layer.linear.lora_down.weight"],
                possible_alpha_patterns=["proj_out.alpha", "final_layer.linear.alpha"],
            ),
        ])

        # Double blocks
        targets.extend([
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=["transformer_blocks.{block}.attn.to_q.lora_B.weight", "transformer_blocks.{block}.attn.to_q.lora_up.weight", "double_blocks.{block}.img_attn.qkv.lora_B.weight", "double_blocks.{block}.img_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.to_q.lora_A.weight", "transformer_blocks.{block}.attn.to_q.lora_down.weight", "double_blocks.{block}.img_attn.qkv.lora_A.weight", "double_blocks.{block}.img_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.to_q.alpha", "double_blocks.{block}.img_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=["transformer_blocks.{block}.attn.to_k.lora_B.weight", "transformer_blocks.{block}.attn.to_k.lora_up.weight", "double_blocks.{block}.img_attn.qkv.lora_B.weight", "double_blocks.{block}.img_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.to_k.lora_A.weight", "transformer_blocks.{block}.attn.to_k.lora_down.weight", "double_blocks.{block}.img_attn.qkv.lora_A.weight", "double_blocks.{block}.img_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.to_k.alpha", "double_blocks.{block}.img_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=["transformer_blocks.{block}.attn.to_v.lora_B.weight", "transformer_blocks.{block}.attn.to_v.lora_up.weight", "double_blocks.{block}.img_attn.qkv.lora_B.weight", "double_blocks.{block}.img_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.to_v.lora_A.weight", "transformer_blocks.{block}.attn.to_v.lora_down.weight", "double_blocks.{block}.img_attn.qkv.lora_A.weight", "double_blocks.{block}.img_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.to_v.alpha", "double_blocks.{block}.img_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_out",
                possible_up_patterns=["transformer_blocks.{block}.attn.to_out.lora_B.weight", "transformer_blocks.{block}.attn.to_out.lora_up.weight", "transformer_blocks.{block}.attn.to_out.0.lora_up.weight", "double_blocks.{block}.img_attn.proj.lora_B.weight", "double_blocks.{block}.img_attn.proj.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.to_out.lora_A.weight", "transformer_blocks.{block}.attn.to_out.lora_down.weight", "transformer_blocks.{block}.attn.to_out.0.lora_down.weight", "double_blocks.{block}.img_attn.proj.lora_A.weight", "double_blocks.{block}.img_attn.proj.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.to_out.alpha", "double_blocks.{block}.img_attn.proj.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=["transformer_blocks.{block}.attn.add_q_proj.lora_B.weight", "transformer_blocks.{block}.attn.add_q_proj.lora_up.weight", "double_blocks.{block}.txt_attn.qkv.lora_B.weight", "double_blocks.{block}.txt_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.add_q_proj.lora_A.weight", "transformer_blocks.{block}.attn.add_q_proj.lora_down.weight", "double_blocks.{block}.txt_attn.qkv.lora_A.weight", "double_blocks.{block}.txt_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.add_q_proj.alpha", "double_blocks.{block}.txt_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=["transformer_blocks.{block}.attn.add_k_proj.lora_B.weight", "transformer_blocks.{block}.attn.add_k_proj.lora_up.weight", "double_blocks.{block}.txt_attn.qkv.lora_B.weight", "double_blocks.{block}.txt_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.add_k_proj.lora_A.weight", "transformer_blocks.{block}.attn.add_k_proj.lora_down.weight", "double_blocks.{block}.txt_attn.qkv.lora_A.weight", "double_blocks.{block}.txt_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.add_k_proj.alpha", "double_blocks.{block}.txt_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=["transformer_blocks.{block}.attn.add_v_proj.lora_B.weight", "transformer_blocks.{block}.attn.add_v_proj.lora_up.weight", "double_blocks.{block}.txt_attn.qkv.lora_B.weight", "double_blocks.{block}.txt_attn.qkv.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.add_v_proj.lora_A.weight", "transformer_blocks.{block}.attn.add_v_proj.lora_down.weight", "double_blocks.{block}.txt_attn.qkv.lora_A.weight", "double_blocks.{block}.txt_attn.qkv.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.add_v_proj.alpha", "double_blocks.{block}.txt_attn.qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=["transformer_blocks.{block}.attn.to_add_out.lora_B.weight", "transformer_blocks.{block}.attn.to_add_out.lora_up.weight", "double_blocks.{block}.txt_attn.proj.lora_B.weight", "double_blocks.{block}.txt_attn.proj.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.attn.to_add_out.lora_A.weight", "transformer_blocks.{block}.attn.to_add_out.lora_down.weight", "double_blocks.{block}.txt_attn.proj.lora_A.weight", "double_blocks.{block}.txt_attn.proj.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.attn.to_add_out.alpha", "double_blocks.{block}.txt_attn.proj.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_in",
                possible_up_patterns=["transformer_blocks.{block}.ff.linear_in.lora_B.weight", "transformer_blocks.{block}.ff.linear_in.lora_up.weight", "double_blocks.{block}.img_mlp.0.lora_B.weight", "double_blocks.{block}.img_mlp.0.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.ff.linear_in.lora_A.weight", "transformer_blocks.{block}.ff.linear_in.lora_down.weight", "double_blocks.{block}.img_mlp.0.lora_A.weight", "double_blocks.{block}.img_mlp.0.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.ff.linear_in.alpha", "double_blocks.{block}.img_mlp.0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear_out",
                possible_up_patterns=["transformer_blocks.{block}.ff.linear_out.lora_B.weight", "transformer_blocks.{block}.ff.linear_out.lora_up.weight", "double_blocks.{block}.img_mlp.2.lora_B.weight", "double_blocks.{block}.img_mlp.2.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.ff.linear_out.lora_A.weight", "transformer_blocks.{block}.ff.linear_out.lora_down.weight", "double_blocks.{block}.img_mlp.2.lora_A.weight", "double_blocks.{block}.img_mlp.2.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.ff.linear_out.alpha", "double_blocks.{block}.img_mlp.2.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_in",
                possible_up_patterns=["transformer_blocks.{block}.ff_context.linear_in.lora_B.weight", "transformer_blocks.{block}.ff_context.linear_in.lora_up.weight", "double_blocks.{block}.txt_mlp.0.lora_B.weight", "double_blocks.{block}.txt_mlp.0.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.ff_context.linear_in.lora_A.weight", "transformer_blocks.{block}.ff_context.linear_in.lora_down.weight", "double_blocks.{block}.txt_mlp.0.lora_A.weight", "double_blocks.{block}.txt_mlp.0.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.ff_context.linear_in.alpha", "double_blocks.{block}.txt_mlp.0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear_out",
                possible_up_patterns=["transformer_blocks.{block}.ff_context.linear_out.lora_B.weight", "transformer_blocks.{block}.ff_context.linear_out.lora_up.weight", "double_blocks.{block}.txt_mlp.2.lora_B.weight", "double_blocks.{block}.txt_mlp.2.lora_up.weight"],
                possible_down_patterns=["transformer_blocks.{block}.ff_context.linear_out.lora_A.weight", "transformer_blocks.{block}.ff_context.linear_out.lora_down.weight", "double_blocks.{block}.txt_mlp.2.lora_A.weight", "double_blocks.{block}.txt_mlp.2.lora_down.weight"],
                possible_alpha_patterns=["transformer_blocks.{block}.ff_context.linear_out.alpha", "double_blocks.{block}.txt_mlp.2.alpha"],
            ),
        ])

        # Single blocks
        targets.extend([
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_qkv_mlp_proj",
                possible_up_patterns=[
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_B.weight",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_up.weight",
                    # BFL format: normalizer converts single_blocks.N → single_transformer_blocks.N
                    "single_transformer_blocks.{block}.linear1.lora_B.weight",
                    "single_transformer_blocks.{block}.linear1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_A.weight",
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.lora_down.weight",
                    # BFL format: normalizer converts single_blocks.N → single_transformer_blocks.N
                    "single_transformer_blocks.{block}.linear1.lora_A.weight",
                    "single_transformer_blocks.{block}.linear1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.alpha",
                    "single_transformer_blocks.{block}.linear1.alpha",
                ],
                up_transform=LoraTransforms.pad_flux2_single_linear1_up,
                down_transform=LoraTransforms.pad_flux2_single_linear1_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_out",
                possible_up_patterns=[
                    "single_transformer_blocks.{block}.attn.to_out.lora_B.weight",
                    "single_transformer_blocks.{block}.attn.to_out.lora_up.weight",
                    # BFL format: normalizer converts single_blocks.N → single_transformer_blocks.N
                    "single_transformer_blocks.{block}.linear2.lora_B.weight",
                    "single_transformer_blocks.{block}.linear2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "single_transformer_blocks.{block}.attn.to_out.lora_A.weight",
                    "single_transformer_blocks.{block}.attn.to_out.lora_down.weight",
                    # BFL format: normalizer converts single_blocks.N → single_transformer_blocks.N
                    "single_transformer_blocks.{block}.linear2.lora_A.weight",
                    "single_transformer_blocks.{block}.linear2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "single_transformer_blocks.{block}.attn.to_out.alpha",
                    "single_transformer_blocks.{block}.linear2.alpha",
                ],
                up_transform=LoraTransforms.pad_flux2_single_linear2_up,
                down_transform=LoraTransforms.pad_flux2_single_linear2_down,
            ),
        ])

        return targets

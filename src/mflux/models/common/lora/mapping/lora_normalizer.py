import re


class LoRANormalizer:
    """Normalizes LoRA weight keys from third-party formats to mflux internal format.

    Supported source formats:
      - Diffusers / PEFT:    ``.lora_A.weight``, ``.lora_B.weight``, ``.default.weight``
      - Kohya / AI Toolkit:  ``.lora.up.``, ``.lora.down.``, underscore-separated block names
      - BFL (Black Forest):  ``lora_unet_double_blocks_N_*``, ``single_blocks_N_*``
      - LyCORIS LoKr:        ``.lokr.w1.``, ``.lokr.w2.``, ``.lokr.w1_a.``, etc.

    Normalization order (order matters):
      1. Suffix standardization  (lora_A/lora_B → lora_down/lora_up)
      2. PEFT ``.default.weight`` stripping
      3. LoKr dot-notation → underscore-notation
      4. Prefix stripping  (``base_model.model.``, ``diffusion_model.``, etc.)
      5. Underscore block names → dot notation  (``double_blocks_0_`` → ``double_blocks.0.``)
      6. BFL ``single_blocks.N.`` → ``single_transformer_blocks.N.``
      7. AIToolkit field name normalization  (``img_attn_qkv`` → ``img_attn.qkv``)
    """

    @staticmethod
    def normalize(weights: dict) -> dict:
        normalized_weights = {}
        for key, value in weights.items():
            new_key = key

            # 1. Suffix standardization
            new_key = new_key.replace(".lora.up.", ".lora_up.")
            new_key = new_key.replace(".lora.down.", ".lora_down.")
            new_key = new_key.replace(".lora_A.", ".lora_down.")
            new_key = new_key.replace(".lora_B.", ".lora_up.")

            # 2. PEFT/Diffusers suffix
            new_key = new_key.replace(".default.weight", ".weight")

            # 3. LoKr dot-notation standardization
            new_key = re.sub(r"([a-z0-9])_lokr_", r"\1.lokr_", new_key)
            new_key = re.sub(r"\.lokr\.w1(\.|$)", r".lokr_w1\1", new_key)
            new_key = re.sub(r"\.lokr\.w2(\.|$)", r".lokr_w2\1", new_key)
            new_key = re.sub(r"\.lokr\.w1_a(\.|$)", r".lokr_w1_a\1", new_key)
            new_key = re.sub(r"\.lokr\.w1_b(\.|$)", r".lokr_w1_b\1", new_key)
            new_key = re.sub(r"\.lokr\.w2_a(\.|$)", r".lokr_w2_a\1", new_key)
            new_key = re.sub(r"\.lokr\.w2_b(\.|$)", r".lokr_w2_b\1", new_key)
            new_key = re.sub(r"\.lokr\.t2(\.|$)", r".lokr_t2\1", new_key)
            new_key = re.sub(r"\.lokr\.alpha(\.|$)", r".alpha\1", new_key)

            # 4. Prefix stripping
            prefixes_to_strip = [
                "base_model.model.",
                "diffusion_model.",
                "transformer.",
                "lora_unet_",
            ]
            for prefix in prefixes_to_strip:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            # 5. Underscore-separated block names → dot notation
            if "double_blocks_" in new_key:
                new_key = re.sub(r"double_blocks_(\d+)_", r"double_blocks.\1.", new_key)
            if "single_blocks_" in new_key:
                new_key = re.sub(r"single_blocks_(\d+)_", r"single_blocks.\1.", new_key)

            # 6. BFL single_blocks → single_transformer_blocks (Flux2 naming)
            if "single_blocks." in new_key:
                new_key = re.sub(r"\bsingle_blocks\.(\d+)\.", r"single_transformer_blocks.\1.", new_key)

            # 7. AIToolkit / Kohya field name normalization
            new_key = new_key.replace("img_attn_qkv", "img_attn.qkv")
            new_key = new_key.replace("img_attn_proj", "img_attn.proj")
            new_key = new_key.replace("txt_attn_qkv", "txt_attn.qkv")
            new_key = new_key.replace("txt_attn_proj", "txt_attn.proj")
            new_key = new_key.replace("img_mlp_0", "img_mlp.0")
            new_key = new_key.replace("img_mlp_2", "img_mlp.2")
            new_key = new_key.replace("txt_mlp_0", "txt_mlp.0")
            new_key = new_key.replace("txt_mlp_2", "txt_mlp.2")

            normalized_weights[new_key] = value

        return normalized_weights

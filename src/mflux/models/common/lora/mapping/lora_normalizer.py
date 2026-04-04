import re

class LoRANormalizer:
    @staticmethod
    def normalize(weights: dict) -> dict:
        normalized_weights = {}
        for key, value in weights.items():
            new_key = key

            # Suffix Standardization
            new_key = new_key.replace(".lora.up.", ".lora_up.")
            new_key = new_key.replace(".lora.down.", ".lora_down.")
            new_key = new_key.replace(".lora_A.", ".lora_down.")
            new_key = new_key.replace(".lora_B.", ".lora_up.")
            
            # PEFT/Diffusers suffix
            new_key = new_key.replace(".default.weight", ".weight")

            # Lokr Standardization
            new_key = new_key.replace(".lokr.w1.", ".lokr_w1.")
            new_key = new_key.replace(".lokr.w2.", ".lokr_w2.")
            new_key = new_key.replace(".lokr.w1_a.", ".lokr_w1_a.")
            new_key = new_key.replace(".lokr.w1_b.", ".lokr_w1_b.")
            new_key = new_key.replace(".lokr.w2_a.", ".lokr_w2_a.")
            new_key = new_key.replace(".lokr.w2_b.", ".lokr_w2_b.")
            new_key = new_key.replace(".lokr.t2.", ".lokr_t2.")
            new_key = new_key.replace(".lokr.alpha", ".alpha")
            
            # Prefix Stripping
            prefixes_to_strip = [
                "base_model.model.",
                "diffusion_model.",
                "transformer.",
                "lora_unet_",
            ]
            for prefix in prefixes_to_strip:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            # Handle underscore-separated block naming (e.g., double_blocks_0_img_attn_qkv)
            if "double_blocks_" in new_key:
                new_key = re.sub(r"double_blocks_(\d+)_", r"double_blocks.\1.", new_key)
            if "single_blocks_" in new_key:
                new_key = re.sub(r"single_blocks_(\d+)_", r"single_blocks.\1.", new_key)

            # AIToolkit / Kohya middle-underscore standardization
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

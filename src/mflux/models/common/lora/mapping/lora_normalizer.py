class LoRANormalizer:
    @staticmethod
    def normalize(weights: dict) -> dict:
        normalized_weights = {}
        for key, value in weights.items():
            new_key = key

            # Suffix Standardization
            new_key = new_key.replace(".lora.up.", ".lora_up.")
            new_key = new_key.replace(".lora.down.", ".lora_down.")
            
            # While AIToolkit often uses lora_A and lora_B, some diffusers use them too.
            # We don't necessarily have to rename them to lora_down/lora_up since mflux mapping 
            # supports lora_A/lora_B directly, but we can if we want to reduce patterns.
            # However, we promised "Normalize lora.up, lora.down, lora_A, lora_B" in the plan.
            new_key = new_key.replace(".lora_A.", ".lora_down.")
            new_key = new_key.replace(".lora_B.", ".lora_up.")

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
            if new_key.startswith("base_model.model."):
                new_key = new_key[len("base_model.model."):]
            if new_key.startswith("diffusion_model."):
                new_key = new_key[len("diffusion_model."):]
            if new_key.startswith("transformer."):
                new_key = new_key[len("transformer."):]

            # Block Translations
            # AIToolkit / diffusers sometimes use "single_blocks" instead of "single_transformer_blocks"
            new_key = new_key.replace("single_blocks", "single_transformer_blocks")

            # AIToolkit Flux2 Edge Case (Translating double_blocks to single_transformer_blocks)
            # AIToolkit trains Flux2 as if it has double blocks (img_attn) 
            # mflux Flux2 uses single_transformer_blocks.attn.to_qkv_mlp_proj
            if "double_blocks" in new_key and "img_attn.qkv" in new_key:
                new_key = new_key.replace("double_blocks", "single_transformer_blocks")
                new_key = new_key.replace("img_attn.qkv", "attn.to_qkv_mlp_proj")
            
            if "double_blocks" in new_key and "img_mlp.0" in new_key:
                new_key = new_key.replace("double_blocks", "single_transformer_blocks")
                new_key = new_key.replace("img_mlp.0", "ff.linear_in")
            
            if "double_blocks" in new_key and "img_mlp.2" in new_key:
                new_key = new_key.replace("double_blocks", "single_transformer_blocks")
                new_key = new_key.replace("img_mlp.2", "ff.linear_out")

            if "double_blocks" in new_key and "img_attn.proj" in new_key:
                new_key = new_key.replace("double_blocks", "single_transformer_blocks")
                new_key = new_key.replace("img_attn.proj", "attn.to_out")

            normalized_weights[new_key] = value
        
        return normalized_weights

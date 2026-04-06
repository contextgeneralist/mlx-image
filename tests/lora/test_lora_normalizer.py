import unittest

from mflux.models.common.lora.mapping.lora_normalizer import LoRANormalizer


class TestLoRANormalizer(unittest.TestCase):
    def test_suffix_standardization(self):
        weights = {
            "layer1.lora.up.weight": 1,
            "layer2.lora.down.weight": 2,
            "layer3.lora_A.weight": 3,
            "layer4.lora_B.weight": 4,
        }
        normalized = LoRANormalizer.normalize(weights)
        self.assertIn("layer1.lora_up.weight", normalized)
        self.assertIn("layer2.lora_down.weight", normalized)
        self.assertIn("layer3.lora_down.weight", normalized)
        self.assertIn("layer4.lora_up.weight", normalized)

    def test_prefix_stripping(self):
        weights = {
            "base_model.model.layer1.weight": 1,
            "diffusion_model.layer2.weight": 2,
            "transformer.layer3.weight": 3,
        }
        normalized = LoRANormalizer.normalize(weights)
        self.assertIn("layer1.weight", normalized)
        self.assertIn("layer2.weight", normalized)
        self.assertIn("layer3.weight", normalized)

    def test_block_translations(self):
        weights = {
            "single_blocks.0.linear1.weight": 1,
        }
        normalized = LoRANormalizer.normalize(weights)
        self.assertIn("single_transformer_blocks.0.linear1.weight", normalized)

    def test_aitoolkit_flux2_edge_case(self):
        weights = {
            # Kohya/AIToolkit uses underscore-separated block names and lora_unet_ prefix
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": 1,
            "lora_unet_double_blocks_1_img_mlp_0.lora_up.weight": 2,
            "lora_unet_double_blocks_2_img_mlp_2.lora_down.weight": 3,
            "lora_unet_double_blocks_3_img_attn_proj.lora_up.weight": 4,
            # BFL single_blocks naming → should become single_transformer_blocks
            "lora_unet_single_blocks_0_linear1.lora_down.weight": 5,
        }
        normalized = LoRANormalizer.normalize(weights)

        # double_blocks remain double_blocks after normalization
        self.assertIn("double_blocks.0.img_attn.qkv.lora_down.weight", normalized)
        self.assertIn("double_blocks.1.img_mlp.0.lora_up.weight", normalized)
        self.assertIn("double_blocks.2.img_mlp.2.lora_down.weight", normalized)
        self.assertIn("double_blocks.3.img_attn.proj.lora_up.weight", normalized)

        # single_blocks → single_transformer_blocks
        self.assertIn("single_transformer_blocks.0.linear1.lora_down.weight", normalized)
        
if __name__ == "__main__":
    unittest.main()

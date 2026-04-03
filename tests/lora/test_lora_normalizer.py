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
            "transformer.double_blocks.0.img_attn.qkv.lora_A.weight": 1,
            "diffusion_model.double_blocks.1.img_mlp.0.lora_B.weight": 2,
            "double_blocks.2.img_mlp.2.lora.down.weight": 3,
            "double_blocks.3.img_attn.proj.lora.up.weight": 4,
        }
        normalized = LoRANormalizer.normalize(weights)
        
        # Test qkv mapping and suffix mapping
        self.assertIn("single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_down.weight", normalized)
        
        # Test mlp mapping and suffix mapping
        self.assertIn("single_transformer_blocks.1.ff.linear_in.lora_up.weight", normalized)
        self.assertIn("single_transformer_blocks.2.ff.linear_out.lora_down.weight", normalized)
        
        # Test proj mapping and suffix mapping
        self.assertIn("single_transformer_blocks.3.attn.to_out.lora_up.weight", normalized)
        
if __name__ == "__main__":
    unittest.main()

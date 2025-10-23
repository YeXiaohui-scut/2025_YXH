"""
Unit tests for semantic watermarking modules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_rotation_matrix():
    """Test rotation matrix encryption/decryption"""
    print("Testing Rotation Matrix...")
    from models.semanticEmbedding import RotationMatrix
    
    dim = 512
    rotation = RotationMatrix(dim=dim, seed=42)
    
    # Test encryption and decryption
    original = torch.randn(4, dim)
    encrypted = rotation.encrypt(original)
    decrypted = rotation.decrypt(encrypted)
    
    # Check dimensions
    assert encrypted.shape == original.shape, "Encrypted shape mismatch"
    assert decrypted.shape == original.shape, "Decrypted shape mismatch"
    
    # Check decryption accuracy
    diff = torch.abs(original - decrypted).max()
    assert diff < 1e-5, f"Decryption error too large: {diff}"
    
    print(f"  ✓ Rotation matrix test passed (max diff: {diff:.2e})")
    return True


def test_semantic_encoder():
    """Test semantic encoder with fallback mode"""
    print("Testing Semantic Encoder...")
    from models.semanticEmbedding import SemanticEncoder
    
    # Initialize with fallback (no CLIP download)
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=512)
    
    # Test encoding
    prompts = ["A photo of an astronaut", "A cat on the moon"]
    semantic_vectors = encoder.encode_text(prompts)
    
    # Check shape
    assert semantic_vectors.shape == (2, 512), f"Shape mismatch: {semantic_vectors.shape}"
    
    # Test determinism
    semantic_vectors2 = encoder.encode_text(prompts)
    diff = torch.abs(semantic_vectors - semantic_vectors2).max()
    assert diff < 1e-5, "Encoding should be deterministic"
    
    # Test encryption
    encrypted = encoder(prompts, encrypt=True)
    assert encrypted.shape == semantic_vectors.shape, "Encrypted shape mismatch"
    
    # Test decryption and verification (use single prompt for verification)
    decrypted = encoder.rotation_matrix.decrypt(encrypted)
    decrypted_single = decrypted[0:1]  # Take first item only
    is_authentic, similarity = encoder.verify(decrypted_single, prompts[0])
    print(f"  ✓ Semantic encoder test passed (similarity: {similarity:.4f})")
    
    return True


def test_unet_wemb():
    """Test U-Net WEmb module"""
    print("Testing U-Net WEmb...")
    from models.unetWEmb import UNetWEmb, LightweightUNetWEmb
    
    semantic_dim = 512
    feature_channels = 256
    batch_size = 2
    h, w = 32, 32
    
    # Test full U-Net
    wemb = UNetWEmb(semantic_dim=semantic_dim, feature_channels=feature_channels, base_channels=32)
    semantic_vector = torch.randn(batch_size, semantic_dim)
    feature_map = torch.randn(batch_size, feature_channels, h, w)
    
    perturbation = wemb(semantic_vector, feature_map)
    
    # Check output shape
    assert perturbation.shape == feature_map.shape, f"Shape mismatch: {perturbation.shape} vs {feature_map.shape}"
    
    # Test lightweight version
    light_wemb = LightweightUNetWEmb(semantic_dim=semantic_dim, feature_channels=feature_channels, base_channels=32)
    perturbation_light = light_wemb(semantic_vector, feature_map)
    
    assert perturbation_light.shape == feature_map.shape, "Lightweight shape mismatch"
    
    print(f"  ✓ U-Net WEmb test passed")
    return True


def test_semantic_decoder():
    """Test semantic decoder"""
    print("Testing Semantic Decoder...")
    from models.semanticDecoder import SemanticDecoder, LightweightSemanticDecoder
    
    semantic_dim = 512
    batch_size = 2
    
    # Test standard decoder
    decoder = SemanticDecoder(semantic_dim=semantic_dim)
    image = torch.randn(batch_size, 3, 256, 256)
    output = decoder(image)
    
    assert output.shape == (batch_size, semantic_dim), f"Output shape mismatch: {output.shape}"
    
    # Test lightweight decoder
    light_decoder = LightweightSemanticDecoder(semantic_dim=semantic_dim)
    output_light = light_decoder(image)
    
    assert output_light.shape == (batch_size, semantic_dim), "Lightweight output shape mismatch"
    
    print(f"  ✓ Semantic decoder test passed")
    return True


def test_integration():
    """Test integration of all components"""
    print("Testing Integration...")
    from models.semanticEmbedding import SemanticEncoder
    from models.unetWEmb import LightweightUNetWEmb
    from models.semanticDecoder import LightweightSemanticDecoder
    
    # Setup
    semantic_dim = 512
    batch_size = 2
    
    # Encoder
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=semantic_dim)
    prompts = ["Test prompt 1", "Test prompt 2"]
    
    # Encode and encrypt
    semantic_vectors = encoder(prompts, encrypt=True)
    
    # WEmb - generate perturbation
    wemb = LightweightUNetWEmb(semantic_dim=semantic_dim, feature_channels=256, base_channels=32)
    feature_map = torch.randn(batch_size, 256, 32, 32)
    perturbation = wemb(semantic_vectors, feature_map)
    
    # Apply perturbation
    watermarked_features = feature_map + perturbation * 0.1
    
    # Simulate image generation (just a dummy upsampling)
    watermarked_image = torch.nn.functional.interpolate(
        watermarked_features, size=(256, 256), mode='bilinear'
    )
    # Convert to 3 channels
    watermarked_image = watermarked_image[:, :3, :, :]
    
    # Extract watermark
    decoder = LightweightSemanticDecoder(semantic_dim=semantic_dim)
    extracted_semantic = decoder(watermarked_image)
    
    # Decrypt and verify
    decrypted = encoder.rotation_matrix.decrypt(extracted_semantic)
    original_semantic = encoder.rotation_matrix.decrypt(semantic_vectors)
    
    # Check similarity
    similarity = torch.nn.functional.cosine_similarity(decrypted, original_semantic, dim=-1)
    print(f"  ✓ Integration test passed (avg similarity: {similarity.mean():.4f})")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Semantic Watermarking Module Tests")
    print("=" * 60)
    
    tests = [
        ("Rotation Matrix", test_rotation_matrix),
        ("Semantic Encoder", test_semantic_encoder),
        ("U-Net WEmb", test_unet_wemb),
        ("Semantic Decoder", test_semantic_decoder),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Tests Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

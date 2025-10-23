"""
Demo script for semantic watermarking
Shows basic usage of the semantic watermarking system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.semanticEmbedding import SemanticEncoder, RotationMatrix
from models.unetWEmb import LightweightUNetWEmb
from models.semanticDecoder import LightweightSemanticDecoder


def demo_rotation_matrix():
    """Demonstrate rotation matrix encryption"""
    print("="*60)
    print("Demo 1: Rotation Matrix Encryption")
    print("="*60)
    
    # Create rotation matrix with seed for reproducibility
    rotation = RotationMatrix(dim=512, seed=42)
    
    # Create sample semantic vector
    original_vector = torch.randn(1, 512)
    print(f"Original vector norm: {original_vector.norm():.4f}")
    
    # Encrypt
    encrypted = rotation.encrypt(original_vector)
    print(f"Encrypted vector norm: {encrypted.norm():.4f}")
    
    # Decrypt
    decrypted = rotation.decrypt(encrypted)
    print(f"Decrypted vector norm: {decrypted.norm():.4f}")
    
    # Verify perfect reconstruction
    reconstruction_error = (original_vector - decrypted).abs().max()
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    print("\n✓ Rotation matrix preserves vector norms and enables perfect reconstruction\n")


def demo_semantic_encoding():
    """Demonstrate semantic encoding from text"""
    print("="*60)
    print("Demo 2: Semantic Text Encoding")
    print("="*60)
    
    # Initialize encoder (will use fallback mode without CLIP)
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=512)
    
    # Test with different prompts
    prompts = [
        "A photo of an astronaut riding a horse on the moon",
        "A beautiful sunset over the ocean",
        "A cat sleeping on a couch"
    ]
    
    print("Encoding prompts...")
    for i, prompt in enumerate(prompts):
        # Encode without encryption
        semantic_vec = encoder.encode_text(prompt)
        print(f"{i+1}. '{prompt}'")
        print(f"   Vector shape: {semantic_vec.shape}, Norm: {semantic_vec.norm():.4f}")
    
    # Test with metadata
    print("\nEncoding with metadata...")
    prompt = prompts[0]
    metadata = {
        'model_version': 'v1.0',
        'user_id': '12345',
        'timestamp': '2025-10-22'
    }
    augmented_prompt = encoder.augment_prompt(prompt, metadata)
    print(f"Original: '{prompt}'")
    print(f"Augmented: '{augmented_prompt}'")
    
    # Encode with encryption
    encrypted_vec = encoder(prompt, encrypt=True, metadata=metadata)
    print(f"Encrypted vector shape: {encrypted_vec.shape}")
    
    print("\n✓ Semantic encoder converts text to high-dimensional vectors\n")


def demo_watermark_embedding():
    """Demonstrate watermark embedding with U-Net"""
    print("="*60)
    print("Demo 3: Watermark Embedding with U-Net")
    print("="*60)
    
    # Create semantic vector
    semantic_dim = 512
    batch_size = 2
    semantic_vector = torch.randn(batch_size, semantic_dim)
    
    # Create feature map (simulating VAE decoder intermediate layer)
    feature_channels = 256
    h, w = 32, 32
    feature_map = torch.randn(batch_size, feature_channels, h, w)
    
    print(f"Input semantic vector shape: {semantic_vector.shape}")
    print(f"Input feature map shape: {feature_map.shape}")
    
    # Create U-Net WEmb module
    wemb = LightweightUNetWEmb(
        semantic_dim=semantic_dim,
        feature_channels=feature_channels,
        base_channels=32
    )
    
    # Generate perturbation
    perturbation = wemb(semantic_vector, feature_map)
    
    print(f"Output perturbation shape: {perturbation.shape}")
    print(f"Perturbation statistics:")
    print(f"  Mean: {perturbation.mean():.6f}")
    print(f"  Std: {perturbation.std():.6f}")
    print(f"  Min: {perturbation.min():.6f}")
    print(f"  Max: {perturbation.max():.6f}")
    
    # Apply watermark
    watermark_strength = 0.1
    watermarked_features = feature_map + perturbation * watermark_strength
    
    # Measure difference
    diff = (watermarked_features - feature_map).abs().mean()
    print(f"\nAverage difference after watermarking: {diff:.6f}")
    
    print("\n✓ U-Net WEmb generates content-adaptive perturbations\n")


def demo_watermark_extraction():
    """Demonstrate watermark extraction and verification"""
    print("="*60)
    print("Demo 4: Watermark Extraction and Verification")
    print("="*60)
    
    # Setup
    semantic_dim = 512
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=semantic_dim)
    decoder = LightweightSemanticDecoder(semantic_dim=semantic_dim)
    
    # Original prompt
    prompt = "A photo of an astronaut riding a horse on the moon"
    print(f"Original prompt: '{prompt}'")
    
    # Encode and encrypt
    semantic_vector = encoder(prompt, encrypt=True)
    print(f"Encoded and encrypted semantic vector: {semantic_vector.shape}")
    
    # Simulate image with watermark (using random image for demo)
    watermarked_image = torch.randn(1, 3, 256, 256)
    
    # Extract watermark
    extracted_semantic = decoder(watermarked_image)
    print(f"Extracted semantic vector: {extracted_semantic.shape}")
    
    # Decrypt
    decrypted_semantic = encoder.rotation_matrix.decrypt(extracted_semantic)
    
    # Verify
    is_authentic, similarity = encoder.verify(decrypted_semantic, prompt)
    
    print(f"\nVerification Results:")
    print(f"  Authentic: {is_authentic}")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"  Threshold: 0.85")
    
    # Test with wrong prompt
    wrong_prompt = "A different prompt"
    is_authentic_wrong, similarity_wrong = encoder.verify(decrypted_semantic, wrong_prompt)
    
    print(f"\nVerification with wrong prompt:")
    print(f"  Authentic: {is_authentic_wrong}")
    print(f"  Cosine Similarity: {similarity_wrong:.4f}")
    
    print("\n✓ Watermark can be extracted and verified against original prompt\n")


def demo_end_to_end():
    """End-to-end demo of the complete watermarking pipeline"""
    print("="*60)
    print("Demo 5: End-to-End Watermarking Pipeline")
    print("="*60)
    
    # Components
    semantic_dim = 512
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=semantic_dim)
    wemb = LightweightUNetWEmb(semantic_dim=semantic_dim, feature_channels=256, base_channels=32)
    decoder = LightweightSemanticDecoder(semantic_dim=semantic_dim)
    
    # 1. Generate semantic watermark
    prompt = "A beautiful mountain landscape at sunset"
    print(f"1. Original prompt: '{prompt}'")
    
    metadata = {'model_version': 'v1.0', 'timestamp': '2025-10-22'}
    semantic_vector = encoder(prompt, encrypt=True, metadata=metadata)
    print(f"2. Generated encrypted semantic vector: {semantic_vector.shape}")
    
    # 2. Embed watermark (simulated)
    feature_map = torch.randn(1, 256, 32, 32)
    perturbation = wemb(semantic_vector, feature_map)
    watermarked_features = feature_map + perturbation * 0.1
    print(f"3. Generated watermark perturbation: {perturbation.shape}")
    
    # 3. Simulate final image
    watermarked_image = torch.nn.functional.interpolate(
        watermarked_features, size=(256, 256), mode='bilinear'
    )
    watermarked_image = watermarked_image[:, :3, :, :]  # Take first 3 channels
    print(f"4. Watermarked image: {watermarked_image.shape}")
    
    # 4. Extract and verify
    extracted = decoder(watermarked_image)
    decrypted = encoder.rotation_matrix.decrypt(extracted)
    is_authentic, similarity = encoder.verify(decrypted, prompt, metadata=metadata)
    
    print(f"5. Verification: Authentic={is_authentic}, Similarity={similarity:.4f}")
    
    print("\n✓ Complete pipeline: Prompt → Semantic Vector → Watermark → Verify\n")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("SEMANTIC WATERMARKING SYSTEM DEMO")
    print("="*60 + "\n")
    
    demos = [
        demo_rotation_matrix,
        demo_semantic_encoding,
        demo_watermark_embedding,
        demo_watermark_extraction,
        demo_end_to_end,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"✗ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor more information, see SEMANTIC_WATERMARKING.md")


if __name__ == "__main__":
    main()

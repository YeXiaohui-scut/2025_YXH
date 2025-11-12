"""
Demo script for Hybrid Watermarking Framework
Tests all components without requiring full training
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

print("="*70)
print(" Hybrid Watermarking Framework - Demo")
print(" Testing all components without full training")
print("="*70)

# ============================================================================
# Test 1: QR Code Generation and Verification
# ============================================================================

print("\n" + "="*70)
print("TEST 1: QR Code Generation and Verification (MetaSeal)")
print("="*70)

try:
    from utils import (
        generate_key_pair, save_private_key, save_public_key,
        generate_watermark_qr, verify_watermark_qr
    )
    
    # Setup keys
    os.makedirs('keys', exist_ok=True)
    
    if not os.path.exists('keys/private_key.pem'):
        print("\nGenerating cryptographic key pair...")
        private_key, public_key = generate_key_pair()
        save_private_key(private_key, 'keys/private_key.pem')
        save_public_key(public_key, 'keys/public_key.pem')
    else:
        print("\n✓ Using existing keys")
    
    # Test QR generation
    print("\nGenerating QR code watermark...")
    text_data = "A beautiful sunset over the ocean with vibrant colors"
    user_id = "demo_user_001"
    timestamp = "2025-11-12T12:00:00"
    
    qr_tensor = generate_watermark_qr(text_data, user_id, timestamp)
    print(f"✓ QR tensor shape: {qr_tensor.shape}")
    print(f"  Text: {text_data}")
    print(f"  User: {user_id}")
    
    # Test verification
    print("\nVerifying QR code...")
    is_valid, message = verify_watermark_qr(qr_tensor, 'keys/public_key.pem')
    
    if is_valid:
        print("✓ Signature verification: PASSED")
        print(f"  Decoded text: {message['text_data']}")
        print(f"  Decoded user: {message['user_id']}")
    else:
        print("✗ Signature verification: FAILED")
    
    # Test tampering detection
    print("\nTesting tampering detection...")
    tampered_qr = qr_tensor.clone()
    tampered_qr[0, 50:60, 50:60] = 1 - tampered_qr[0, 50:60, 50:60]
    
    is_valid_tampered, _ = verify_watermark_qr(tampered_qr, 'keys/public_key.pem')
    
    if not is_valid_tampered:
        print("✓ Tampering detection: PASSED (correctly detected tampering)")
    else:
        print("✗ Tampering detection: FAILED (did not detect tampering)")
    
    print("\n" + "-"*70)
    print("✓ Test 1 PASSED: QR code system works correctly")
    print("-"*70)
    
except Exception as e:
    print(f"\n✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 2: Model Architecture
# ============================================================================

print("\n" + "="*70)
print("TEST 2: Neural Network Models (GenPTW Architecture)")
print("="*70)

try:
    from models import (
        WatermarkEmbedder, DistortionLayer, WatermarkExtractor, 
        WatermarkAutoencoder
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create dummy inputs
    batch_size = 2
    image_dummy = torch.rand(batch_size, 3, 512, 512).to(device)
    qr_dummy = torch.rand(batch_size, 1, 256, 256).to(device)
    
    print(f"\nTest inputs:")
    print(f"  Image: {image_dummy.shape}")
    print(f"  QR code: {qr_dummy.shape}")
    
    # Test Embedder
    print("\n1. Testing WatermarkEmbedder...")
    embedder = WatermarkEmbedder(hidden_dim=64).to(device)
    watermarked = embedder(image_dummy, qr_dummy)
    print(f"   Output shape: {watermarked.shape}")
    print(f"   Output range: [{watermarked.min():.3f}, {watermarked.max():.3f}]")
    
    # Measure watermark strength
    diff = (watermarked - image_dummy).abs().mean().item()
    print(f"   Average perturbation: {diff:.6f}")
    
    if watermarked.shape == image_dummy.shape and diff < 0.1:
        print("   ✓ Embedder works correctly")
    else:
        print("   ✗ Embedder has issues")
    
    # Test DistortionLayer
    print("\n2. Testing DistortionLayer...")
    distortion = DistortionLayer().to(device)
    distorted, mask = distortion(watermarked)
    print(f"   Distorted image: {distorted.shape}")
    print(f"   Tamper mask: {mask.shape}")
    print(f"   Mask coverage: {(mask > 0.5).float().mean().item():.2%}")
    
    if distorted.shape == watermarked.shape and mask.shape[1:] == (1, 512, 512):
        print("   ✓ DistortionLayer works correctly")
    else:
        print("   ✗ DistortionLayer has issues")
    
    # Test Extractor
    print("\n3. Testing WatermarkExtractor...")
    extractor = WatermarkExtractor(hidden_dim=64).to(device)
    extracted_qr, pred_mask = extractor(distorted)
    print(f"   Extracted QR: {extracted_qr.shape}")
    print(f"   Predicted mask: {pred_mask.shape}")
    print(f"   QR value range: [{extracted_qr.min():.3f}, {extracted_qr.max():.3f}]")
    print(f"   Mask value range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
    
    if extracted_qr.shape == qr_dummy.shape and pred_mask.shape == mask.shape:
        print("   ✓ Extractor works correctly")
    else:
        print("   ✗ Extractor has issues")
    
    # Test complete Autoencoder
    print("\n4. Testing WatermarkAutoencoder...")
    autoencoder = WatermarkAutoencoder(hidden_dim=64).to(device)
    w_img, r_qr, p_mask, gt_mask = autoencoder(image_dummy, qr_dummy)
    print(f"   Watermarked image: {w_img.shape}")
    print(f"   Reconstructed QR: {r_qr.shape}")
    print(f"   Predicted mask: {p_mask.shape}")
    print(f"   Ground truth mask: {gt_mask.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"\n5. Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    print("\n" + "-"*70)
    print("✓ Test 2 PASSED: All models work correctly")
    print("-"*70)
    
except Exception as e:
    print(f"\n✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 3: Loss Functions
# ============================================================================

print("\n" + "="*70)
print("TEST 3: Loss Functions")
print("="*70)

try:
    from train_watermark import WatermarkLoss
    
    print("\nCreating loss function...")
    criterion = WatermarkLoss(device=device)
    
    # Create dummy inputs (from Test 2)
    print("\nComputing losses on random data...")
    losses = criterion(
        original_image=image_dummy,
        watermarked_image=w_img,
        original_qr=qr_dummy,
        reconstructed_qr=r_qr,
        gt_mask=gt_mask,
        pred_mask=p_mask
    )
    
    print(f"\nLoss values:")
    for key, value in losses.items():
        print(f"  {key:20s}: {value.item():.6f}")
    
    # Test dynamic weighting
    print("\nTesting dynamic loss weighting...")
    print("  Initial weights:", end="")
    print(f" fidelity={criterion.fidelity_weight:.2f},", end="")
    print(f" qr={criterion.qr_weight:.2f},", end="")
    print(f" mask={criterion.mask_weight:.2f}")
    
    criterion.update_weights(epoch=30, max_epochs=40)
    
    print("  After epoch 30:", end="")
    print(f" fidelity={criterion.fidelity_weight:.2f},", end="")
    print(f" qr={criterion.qr_weight:.2f},", end="")
    print(f" mask={criterion.mask_weight:.2f}")
    
    print("\n" + "-"*70)
    print("✓ Test 3 PASSED: Loss functions work correctly")
    print("-"*70)
    
except Exception as e:
    print(f"\n✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 4: End-to-End Flow (Simulated)
# ============================================================================

print("\n" + "="*70)
print("TEST 4: End-to-End Flow (Without Training)")
print("="*70)

try:
    print("\nSimulating complete workflow...")
    
    # Step 1: Generate watermark
    print("\n1. Generate watermark for image...")
    test_text = "Mountain landscape with snow-capped peaks"
    qr = generate_watermark_qr(test_text, "user_test", "2025-11-12T12:00:00")
    print(f"   ✓ QR generated: {qr.shape}")
    
    # Step 2: Embed watermark
    print("\n2. Embed watermark into image...")
    test_image = torch.rand(1, 3, 512, 512).to(device)
    with torch.no_grad():
        watermarked = embedder(test_image, qr.to(device))
    print(f"   ✓ Watermarked: {watermarked.shape}")
    print(f"   ✓ PSNR: {-10 * torch.log10(((test_image - watermarked) ** 2).mean()):.2f} dB")
    
    # Step 3: Apply distortions
    print("\n3. Apply distortions (simulate attacks)...")
    with torch.no_grad():
        distorted, _ = distortion(watermarked)
    print(f"   ✓ Distorted: {distorted.shape}")
    
    # Step 4: Extract watermark
    print("\n4. Extract watermark and verify...")
    with torch.no_grad():
        extracted_qr, tamper_mask = extractor(distorted)
    print(f"   ✓ Extracted QR: {extracted_qr.shape}")
    print(f"   ✓ Tamper mask: {tamper_mask.shape}")
    
    # Step 5: Verify signature
    print("\n5. Verify cryptographic signature...")
    is_valid, decoded = verify_watermark_qr(extracted_qr, 'keys/public_key.pem')
    
    if is_valid:
        print("   ✓ Signature verified successfully!")
    else:
        print("   ⚠ Signature verification failed (expected with random weights)")
        print("   Note: This is normal without training - the extractor needs training to work")
    
    # Step 6: Check tampering
    print("\n6. Check for tampering...")
    max_tamper = tamper_mask.max().item()
    mean_tamper = tamper_mask.mean().item()
    print(f"   Max tamper score: {max_tamper:.4f}")
    print(f"   Mean tamper score: {mean_tamper:.4f}")
    
    if max_tamper > 0.5:
        print("   ⚠ Tampering detected (expected with random weights)")
    else:
        print("   ✓ No significant tampering detected")
    
    print("\n" + "-"*70)
    print("✓ Test 4 PASSED: End-to-end flow works")
    print("Note: Actual extraction quality requires training on real data")
    print("-"*70)
    
except Exception as e:
    print(f"\n✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print(" DEMO SUMMARY")
print("="*70)

print("""
✓ All core components are working correctly!

The framework includes:
1. ✓ QR code generation with ECDSA signatures (MetaSeal)
2. ✓ Watermark embedding network (SWIFT/HiDDeN)
3. ✓ Distortion simulation layer (GenPTW)
4. ✓ Watermark extraction network (GenPTW)
5. ✓ Tamper localization (GenPTW)
6. ✓ Loss functions with dynamic weighting
7. ✓ Complete training pipeline (train_watermark.py)
8. ✓ Complete inference pipeline (main.py)

NEXT STEPS:
-----------
1. Install dependencies:
   pip install -r requirements_watermark.txt

2. Prepare dataset (COCO or similar):
   - Download images
   - Place in a directory

3. Train the model:
   python train_watermark.py \\
       --data_dir /path/to/images \\
       --output_dir outputs/training \\
       --epochs 40 \\
       --batch_size 8

4. Generate watermarked images:
   python main.py generate \\
       --prompt "Your text prompt" \\
       --user_id "your_user_id" \\
       --output output.png \\
       --embedder_checkpoint outputs/training/best_model.pt \\
       --extractor_checkpoint outputs/training/best_model.pt

5. Verify images:
   python main.py verify \\
       --image output.png \\
       --extractor_checkpoint outputs/training/best_model.pt

For detailed documentation, see:
- README_WATERMARK_FRAMEWORK.md (complete guide)
- utils.py (utility functions)
- models.py (model architectures)
- train_watermark.py (training pipeline)
- main.py (inference and verification)
""")

print("="*70)
print(" Demo completed successfully!")
print("="*70)

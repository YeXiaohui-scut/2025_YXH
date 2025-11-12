# Hybrid Watermarking Framework

A comprehensive post-hoc watermarking system for AI-generated content that combines three state-of-the-art approaches:

- **MetaSeal**: Cryptographic QR code watermarks with ECDSA signatures
- **SWIFT**: BLIP-based semantic extraction for content-aware watermarking  
- **GenPTW**: Frequency-coordinated extraction and tamper localization

## 🎯 Overview

This framework operates in two stages:

### Stage 1: Training (Autoencoder)
Train on real images (e.g., COCO dataset) to learn robust watermark embedding and extraction:
```
Real Image → [Semantic Extraction] → QR Code Generation → Watermark Embedding
    ↓
Distortion Simulation (AIGC attacks + degradations)
    ↓
Watermark & Tamper Extraction → Loss Computation → Model Update
```

### Stage 2: Inference (Post-Processing)
Apply trained model to T2I generated images:
```
Text Prompt → T2I Model (Stable Diffusion) → Generated Image
    ↓
[Semantic Extraction] → QR Code + Signature → Embed Watermark
    ↓
Watermarked Image (ready for distribution)
```

### Verification
Detect tampering and verify authenticity:
```
Watermarked Image → Extract QR Code + Tamper Mask
    ↓
├─ Verify Cryptographic Signature (MetaSeal)
├─ Check Tamper Localization (GenPTW)
└─ Compare Semantic Consistency (SWIFT)
```

## 📁 File Structure

```
.
├── utils.py                      # Utility functions
│   ├── load_semantic_extractor() # BLIP model loading
│   ├── generate_watermark_qr()   # QR code + ECDSA signing
│   ├── verify_watermark_qr()     # Signature verification
│   └── load_t2i_model()          # Stable Diffusion loading
│
├── models.py                     # Neural network architectures
│   ├── WatermarkEmbedder         # CNN-based watermark embedding
│   ├── DistortionLayer           # Attack simulation
│   ├── WatermarkExtractor        # Frequency-coordinated decoder
│   └── WatermarkAutoencoder      # Complete training model
│
├── train_watermark.py            # Training script (Stage 1)
│   ├── WatermarkTrainingDataset  # Dataset with BLIP extraction
│   ├── WatermarkLoss             # Combined loss functions
│   └── Training loop             # With dynamic loss weighting
│
├── main.py                       # Inference script (Stage 2)
│   ├── WatermarkPipeline         # Complete pipeline
│   ├── generate_and_watermark()  # T2I + watermarking
│   └── verify_image()            # Multi-level verification
│
└── requirements_watermark.txt    # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_watermark.txt

# Generate cryptographic keys (one-time setup)
python -c "from utils import test_qr_generation_and_verification; test_qr_generation_and_verification()"
```

This will create `keys/private_key.pem` and `keys/public_key.pem`.

### 2. Training (Stage 1)

Train the autoencoder on real images:

```bash
python train_watermark.py \
    --data_dir /path/to/coco/images \
    --output_dir outputs/watermark_training \
    --batch_size 8 \
    --epochs 40 \
    --learning_rate 1e-4
```

**Training Details:**
- Dataset: COCO or similar real image dataset
- Duration: ~30-40 epochs (~15-20 hours on A100)
- Automatic semantic extraction using BLIP
- Dynamic loss weighting (GenPTW strategy)
- Saves checkpoints every 5 epochs

**Expected Results after 40 epochs:**
- QR Reconstruction Loss: < 0.01
- Fidelity Loss (MSE): < 0.001
- LPIPS: < 0.05
- Tamper Detection IoU: > 0.85

### 3. Inference (Stage 2)

#### Generate Watermarked Image

```bash
python main.py generate \
    --prompt "A beautiful sunset over the ocean" \
    --user_id "user_12345" \
    --output outputs/watermarked_sunset.png \
    --embedder_checkpoint outputs/watermark_training/best_model.pt \
    --extractor_checkpoint outputs/watermark_training/best_model.pt \
    --private_key keys/private_key.pem
```

#### Verify Image

```bash
python main.py verify \
    --image outputs/watermarked_sunset.png \
    --extractor_checkpoint outputs/watermark_training/best_model.pt \
    --public_key keys/public_key.pem \
    --use_bert  # Optional: use BERT for semantic similarity
```

## 📊 Architecture Details

### WatermarkEmbedder
- CNN-based encoder (inspired by HiDDeN/SWIFT)
- Fuses image features with QR code features
- Generates imperceptible residuals with JND constraint
- Output: Watermarked image with < 40 dB PSNR

### DistortionLayer (Training Only)
Simulates various attacks for robustness:
- **AIGC Attacks:**
  - VAE reconstruction (encode-decode cycle)
  - Watermark region removal (simulated inpainting)
  - Real inpainting editing
- **Common Degradations:**
  - JPEG compression (quality 50-95)
  - Gaussian noise (σ = 0.01-0.05)
  - Brightness/contrast adjustments

### WatermarkExtractor
- **DCT Module**: Separates frequency components
- **Low-Frequency Branch (W_Dec)**: Reconstructs QR code watermark
- **High-Frequency Branch (CN_Enc)**: ConvNeXt-based feature extraction
- **Mask Decoder**: Predicts tamper localization map

Outputs:
1. Reconstructed QR code [1, 256, 256]
2. Tamper mask [1, H, W] (0 = authentic, 1 = tampered)

## 🔐 Security Features

### 1. Cryptographic Signing (MetaSeal)
- ECDSA with P-256 curve
- Signs: `{text_data, user_id, timestamp}`
- QR code contains: `{message, signature}`
- Tamper detection through signature verification

### 2. Semantic Binding (SWIFT)
- BLIP extracts semantic description from image
- Description is embedded in watermark
- Verification compares current vs. original semantics
- Detects semantic manipulation attacks

### 3. Tamper Localization (GenPTW)
- Pixel-level tamper detection
- Edge-aware loss for precise boundaries
- Multi-scale feature fusion
- Distinguishes authentic from edited regions

## 📈 Loss Functions

The framework uses multiple losses with dynamic weighting:

```python
L_total = w_fid * L_fidelity +      # Image quality
          w_lpips * L_lpips +        # Perceptual quality
          w_qr * L_qr +              # QR reconstruction
          w_mask * L_mask_mse +      # Tamper detection
          w_edge * L_mask_edge       # Edge precision
```

**Dynamic Weighting Strategy:**
- Early training (epochs 1-20): Focus on extraction (high w_qr, w_mask)
- Late training (epochs 21-40): Focus on fidelity (high w_fid)

## 🧪 Testing

### Test Individual Components

```bash
# Test utils (QR generation & verification)
python utils.py

# Test models (all neural networks)
python models.py

# Test training pipeline (dry run)
python train_watermark.py --data_dir /path/to/images --epochs 1 --batch_size 2
```

### Expected Test Outputs

**utils.py:**
```
✓ QR code generated and verified
✓ Signature validation works
✓ Tampering detection works
```

**models.py:**
```
✓ WatermarkEmbedder: [2, 3, 512, 512] → [2, 3, 512, 512]
✓ DistortionLayer: [2, 3, 512, 512] → [2, 3, 512, 512] + mask
✓ WatermarkExtractor: [2, 3, 512, 512] → QR [2, 1, 256, 256] + mask [2, 1, 512, 512]
✓ Total parameters: ~8M
```

## 🎓 Example Workflow

### Complete End-to-End Example

```bash
# Step 1: Generate keys (one-time)
python -c "from utils import generate_key_pair, save_private_key, save_public_key; \
           import os; os.makedirs('keys', exist_ok=True); \
           priv, pub = generate_key_pair(); \
           save_private_key(priv, 'keys/private_key.pem'); \
           save_public_key(pub, 'keys/public_key.pem')"

# Step 2: Train model (Stage 1)
python train_watermark.py \
    --data_dir /path/to/coco \
    --output_dir outputs/training \
    --epochs 40 \
    --batch_size 8

# Step 3: Generate watermarked image (Stage 2)
python main.py generate \
    --prompt "A majestic mountain landscape" \
    --user_id "alice" \
    --output outputs/mountain.png \
    --embedder_checkpoint outputs/training/best_model.pt \
    --extractor_checkpoint outputs/training/best_model.pt

# Step 4: Verify image
python main.py verify \
    --image outputs/mountain.png \
    --extractor_checkpoint outputs/training/best_model.pt \
    --use_bert

# Expected output:
# ✓ Signature is VALID
# ✓ No tampering detected
# ✓ Semantic content is consistent
# Final Verdict: AUTHENTIC
```

## 🔍 Verification Examples

### Scenario 1: Authentic Image
```
Tampering Detection: ✓ No tampering (max_score: 0.12)
Signature Verification: ✓ Valid signature
Semantic Consistency: ✓ Similarity: 0.87
Verdict: AUTHENTIC
```

### Scenario 2: Tampered Image
```
Tampering Detection: ⚠ Tampering detected (max_score: 0.89)
Signature Verification: ✓ Valid signature  
Semantic Consistency: ⚠ Similarity: 0.42
Verdict: SUSPICIOUS
```

### Scenario 3: Forged Image
```
Tampering Detection: ✓ No tampering
Signature Verification: ✗ Invalid signature
Semantic Consistency: N/A
Verdict: SUSPICIOUS (Forged watermark)
```

## 📝 Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Model hidden dimension |
| `image_size` | 512 | Input image resolution |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `epochs` | 40 | Training epochs |
| `jnd_strength` | 0.05 | Maximum watermark perturbation |

### Loss Weights (Initial)

| Loss | Weight | Purpose |
|------|--------|---------|
| Fidelity | 1.0 | Image quality preservation |
| LPIPS | 1.0 | Perceptual quality |
| QR Recon | 10.0 | Watermark extraction accuracy |
| Mask MSE | 5.0 | Tamper detection accuracy |
| Mask Edge | 2.0 | Tamper boundary precision |

## 🔧 Troubleshooting

### Issue: BLIP not loading
```
Solution: BLIP is optional. The system will use fallback mode (filename or prompt as semantic).
```

### Issue: Out of memory during training
```
Solution: Reduce batch_size (e.g., --batch_size 4) or hidden_dim (--hidden_dim 32)
```

### Issue: QR code cannot be decoded
```
Possible causes:
1. Too much distortion during training (reduce distortion strength)
2. Not enough training epochs (increase to 50+)
3. QR code resolution too low (already at 256x256, optimal)
```

### Issue: Poor tamper localization
```
Solution: Increase mask_weight in loss function, or train longer
```

## 🚀 Advanced Usage

### Custom Dataset

```python
from train_watermark import WatermarkTrainingDataset

dataset = WatermarkTrainingDataset(
    image_dir="/path/to/images",
    image_size=512,
    semantic_model=model,
    semantic_processor=processor,
    device='cuda'
)
```

### Custom T2I Model

```python
from utils import load_t2i_model

# Load different model
pipe = load_t2i_model("stabilityai/stable-diffusion-2-1", device='cuda')
```

### Batch Verification

```python
from main import WatermarkPipeline

pipeline = WatermarkPipeline(
    embedder_checkpoint="",
    extractor_checkpoint="best_model.pt"
)

for img_path in image_list:
    results = pipeline.verify_image(img_path)
    print(f"{img_path}: {results['verdict']}")
```

## 📚 References

1. **MetaSeal**: "MetaSeal: A Secure Watermarking Framework for AI-Generated Content"
2. **SWIFT**: "SWIFT: Semantic Watermarking for Image Forgery Detection and Traceability"
3. **GenPTW**: "GenPTW: Post-Training Watermarking for AI-Generated Images with Tamper Localization"

## 📄 License

This implementation is for research purposes. Please cite the original papers when using this framework.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better DCT implementation (currently simplified)
- More sophisticated inpainting simulation
- Multi-resolution watermarking
- Video watermarking extension
- Real-time inference optimization

## 💡 Tips

1. **Training Data**: Use diverse, high-quality images (COCO, ImageNet, etc.)
2. **Key Management**: Keep private keys secure; only distribute public keys
3. **Watermark Strength**: Adjust `jnd_strength` based on imperceptibility requirements
4. **Verification**: Always use all three checks (signature, tampering, semantic) for best security

## 🎉 Acknowledgments

This framework combines ideas from three excellent papers. Special thanks to the authors of MetaSeal, SWIFT, and GenPTW for their contributions to the watermarking community.

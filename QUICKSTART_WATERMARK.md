# Quick Start Guide - Hybrid Watermarking Framework

Get started with the watermarking framework in 5 minutes!

## 🎯 What is This?

A post-hoc watermarking system for AI-generated images that combines:
- **MetaSeal**: Cryptographic signatures (anti-forgery)
- **SWIFT**: Semantic watermarks (content-aware)
- **GenPTW**: Tamper detection (localization)

## 🚀 Quick Demo (No Training Required)

Test all components without training:

```bash
# Run the demo script
python demo_watermark.py
```

This will test:
- ✓ QR code generation and verification
- ✓ Cryptographic signing
- ✓ Neural network models
- ✓ Loss functions
- ✓ End-to-end flow

Expected output: All tests should pass!

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements_watermark.txt

# Alternative: Install core packages manually
pip install torch torchvision pillow qrcode pyzbar cryptography transformers lpips opencv-python
```

## 🔑 One-Time Setup: Generate Keys

```bash
python -c "from utils import test_qr_generation_and_verification; test_qr_generation_and_verification()"
```

This creates `keys/private_key.pem` and `keys/public_key.pem`.

## 📚 Three Usage Modes

### Mode 1: Quick Testing (No Training)

Just test the components:

```bash
# Test utilities
python utils.py

# Test models
python models.py

# Full demo
python demo_watermark.py
```

### Mode 2: Full Training (Recommended)

Train on real images for production use:

```bash
# Prepare your dataset (COCO, ImageNet, or custom)
# Directory structure: /path/to/images/*.jpg

# Train the model
python train_watermark.py \
    --data_dir /path/to/images \
    --output_dir outputs/training \
    --batch_size 8 \
    --epochs 40
```

**Training time:** ~15-20 hours on A100 (40 epochs)

**Expected results after training:**
- QR reconstruction: >99% accuracy
- PSNR: >40 dB
- Tamper detection IoU: >85%

### Mode 3: Inference (Use Trained Model)

#### Generate Watermarked Image

```bash
python main.py generate \
    --prompt "A beautiful mountain landscape" \
    --user_id "alice" \
    --output watermarked.png \
    --embedder_checkpoint outputs/training/best_model.pt \
    --extractor_checkpoint outputs/training/best_model.pt
```

#### Verify Image

```bash
python main.py verify \
    --image watermarked.png \
    --extractor_checkpoint outputs/training/best_model.pt \
    --use_bert
```

## 🎓 Complete Workflow Example

```bash
# Step 1: Setup (one-time)
pip install -r requirements_watermark.txt
python demo_watermark.py  # Test everything works

# Step 2: Train (once, ~20 hours)
python train_watermark.py \
    --data_dir /path/to/coco \
    --output_dir outputs/model \
    --epochs 40

# Step 3: Generate watermarked images (fast, ~10 seconds each)
python main.py generate \
    --prompt "Sunset over ocean" \
    --user_id "user001" \
    --output img1.png \
    --embedder_checkpoint outputs/model/best_model.pt \
    --extractor_checkpoint outputs/model/best_model.pt

# Step 4: Verify images (fast, <1 second)
python main.py verify \
    --image img1.png \
    --extractor_checkpoint outputs/model/best_model.pt
```

## 📊 File Overview

| File | Purpose | Size | Lines |
|------|---------|------|-------|
| `utils.py` | QR codes, crypto, BLIP | 15KB | 500+ |
| `models.py` | Neural networks | 20KB | 700+ |
| `train_watermark.py` | Training pipeline | 16KB | 500+ |
| `main.py` | Inference & verify | 15KB | 500+ |
| `demo_watermark.py` | Quick demo | 12KB | 400+ |
| `requirements_watermark.txt` | Dependencies | 1KB | 30 |
| `README_WATERMARK_FRAMEWORK.md` | Full docs | 12KB | 400+ |

## 🔍 Understanding the Output

### Generation Output
```
✓ Image generated
✓ Semantic extracted: "A beautiful mountain landscape with snow"
✓ QR code generated
✓ Watermark embedded
✓ Saved to watermarked.png
```

### Verification Output (Authentic)
```
✓ No tampering detected (max_score: 0.12)
✓ Signature is VALID
✓ Original text: "A beautiful mountain landscape with snow"
✓ Semantic similarity: 0.89
Final Verdict: AUTHENTIC
```

### Verification Output (Tampered)
```
⚠ Tampering detected (max_score: 0.87)
✓ Signature is VALID (watermark present)
⚠ Semantic similarity: 0.34
Final Verdict: SUSPICIOUS
```

## 🐛 Common Issues

### Issue: Module not found

```bash
# Solution: Install missing package
pip install [package-name]
```

### Issue: CUDA out of memory

```bash
# Solution: Reduce batch size
python train_watermark.py --batch_size 4  # or 2
```

### Issue: BLIP not loading

```
# This is OK! System will use fallback mode
# Watermarking still works, just without semantic extraction
```

### Issue: QR code not decoding

```
# This is expected with untrained models
# Train the model first, or use the demo mode
```

## 📖 Next Steps

1. **Read the full documentation:**
   ```bash
   cat README_WATERMARK_FRAMEWORK.md
   ```

2. **Understand the code:**
   - `utils.py` - Start here for utilities
   - `models.py` - Neural network architectures
   - `train_watermark.py` - Training pipeline
   - `main.py` - Inference and verification

3. **Experiment:**
   - Try different prompts
   - Test with different attacks
   - Adjust watermark strength
   - Fine-tune loss weights

## 💡 Key Concepts

### Two-Stage Design
1. **Stage 1 (Training):** Learn to embed/extract watermarks robustly
2. **Stage 2 (Inference):** Apply to T2I-generated images

### Three Security Levels
1. **Cryptographic:** ECDSA signature (can't be forged)
2. **Semantic:** Content description (detects semantic changes)
3. **Spatial:** Pixel-level tamper detection (locates edits)

### Loss Functions
- **Fidelity:** Keep image quality high
- **LPIPS:** Maintain perceptual quality
- **QR Reconstruction:** Extract watermark accurately
- **Tamper Detection:** Locate edited regions precisely

## 🎉 Success Checklist

After following this guide, you should be able to:

- [ ] Run demo and see all tests pass
- [ ] Understand the two-stage workflow
- [ ] Generate cryptographic keys
- [ ] Train a model (optional but recommended)
- [ ] Generate watermarked images
- [ ] Verify image authenticity
- [ ] Detect tampering and forgery

## 🤝 Need Help?

1. Check `README_WATERMARK_FRAMEWORK.md` for detailed documentation
2. Run `demo_watermark.py` to test your setup
3. Read the code comments in `utils.py`, `models.py`, etc.

## 📚 References

- **MetaSeal**: Cryptographic watermarking with signatures
- **SWIFT**: Semantic watermarking with BLIP
- **GenPTW**: Post-training watermarking with tamper localization

---

**Ready to start?** Run `python demo_watermark.py` now! 🚀

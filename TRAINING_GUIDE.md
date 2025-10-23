# Semantic Watermarking Training Guide

This guide explains how to train the semantic watermarking model (Wemb module) and integrate text prompts for content-related watermarks.

## Overview

The semantic watermarking system embeds watermarks based on text prompts. Each image can have its own unique, content-related watermark derived from its description or caption.

## Training Workflow

### Phase 1: Training with Images (Wemb Module)

In the first phase, you train the Wemb (Watermark Embedding) module using **images with generic or random prompts**. This establishes the basic watermarking capability.

```bash
python train.py \
    --config configs/SD14_SemanticLaWa.yaml \
    --batch_size 8 \
    --max_epochs 40 \
    --learning_rate 0.00006 \
    --output results/semantic_lawa_training
```

**What happens during training:**

1. **Image Loading**: Images are loaded from your dataset
2. **Prompt Assignment**: Each image gets a prompt using one of these strategies:
   - **Generic prompts** (default): Random selections from a pool of 100+ generic descriptions
   - **Real captions** (if available): Load from a caption file
   - **Hash-based assignment**: Each image path gets a consistent prompt within an epoch

3. **Semantic Vector Generation**: 
   - The prompt is encoded into a 512-dimensional semantic vector using CLIP text encoder
   - The vector is encrypted using a rotation matrix (cryptographic security)

4. **Watermark Embedding**:
   - The Wemb U-Net modules generate content-adaptive perturbations
   - Perturbations are injected at 6 strategic points in the VAE decoder
   - The result is a watermarked image that looks identical to the original

5. **Loss Optimization**:
   ```python
   total_loss = (
       recon_weight * reconstruction_loss +      # Image quality
       adversarial_weight * discriminator_loss + # Realism
       semantic_weight * (1 - cosine_similarity) # Watermark accuracy
   )
   ```

### Phase 2: Using Captions (Optional)

If you have image captions, you can use them for training:

#### Option A: Prepare Caption File

Create a CSV file with image paths and captions:

```csv
path,caption
image_001.jpg,A beautiful sunset over the ocean
image_002.jpg,A cat sitting on a window sill
image_003.jpg,A plate of delicious pasta
```

#### Option B: Update Config

Modify your config to use the caption file:

```yaml
data:
  target: tools.dataset.DataModuleWithPrompts
  params:
    train:
      target: tools.dataset.datasetWithPrompts
      params:
        data_dir: /data/mirflickr1m/images
        data_list: data/train_100k.csv
        caption_file: data/train_captions.csv  # Add this
        resize: 256
```

### Phase 3: Text-to-Image (T2I) Integration

After training, you can use the model for T2I generation with content-specific watermarks:

```python
from models.semanticLaWa import SemanticLaWa
from models.semanticEmbedding import SemanticEncoder

# Load trained model
model = SemanticLaWa.load_from_checkpoint('path/to/checkpoint.ckpt')
encoder = model.semantic_encoder

# Generate with specific prompt
prompt = "A white plate of food on a dining table"
metadata = {
    'model_version': 'v1.0',
    'user_id': '12345',
    'timestamp': '2025-10-23'
}

# Encode prompt to semantic vector
semantic_vector = encoder(prompt, encrypt=True, metadata=metadata)

# Generate watermarked image
# (Assuming you have latent from Stable Diffusion)
watermarked_image = model(latent, original_image, semantic_vector)
```

**Each image gets a unique watermark based on:**
- The specific text prompt used to generate it
- Optional metadata (model version, user ID, timestamp)
- Content-adaptive perturbations from Wemb module

## Understanding the Components

### 1. Wemb Module (U-Net)

The Wemb module learns to generate **content-adaptive perturbations**:

```
Input: Semantic Vector (512-dim) + Feature Map (from VAE decoder)
       ↓
Output: Perturbation Pattern (same size as feature map)
```

**Why U-Net architecture?**
- Processes feature maps at the same resolution as VAE decoder
- Skip connections preserve image details
- Content-adaptive: perturbations match image textures
- Multi-scale: different modules for different decoder layers

**Training objective:**
- Generate perturbations that embed the semantic vector
- Keep perturbations invisible (minimize reconstruction loss)
- Make perturbations robust to attacks (adversarial loss)

### 2. Semantic Encoder

Converts text prompts to semantic vectors:

```python
prompt = "A beautiful landscape"
        ↓ (CLIP Text Encoder)
semantic_vector = [0.23, -0.45, 0.12, ...]  # 512 dimensions
        ↓ (Rotation Matrix Encryption)
encrypted_vector = [0.34, 0.21, -0.67, ...]  # Secure
```

### 3. Semantic Decoder

Extracts semantic vectors from watermarked images:

```python
watermarked_image → ResNet50 → Extracted Vector (512-dim)
```

## Training Strategies

### Strategy 1: Generic Prompts (Recommended for Start)

**Best for:** Initial training without captions

```yaml
# Uses default prompt pool (100+ generic prompts)
# No caption file needed
train:
  target: tools.dataset.datasetWithPrompts
  params:
    data_dir: /data/images
    data_list: data/train.csv
    resize: 256
    # No caption_file specified
```

**Advantages:**
- ✅ Simple setup - no caption data needed
- ✅ Works with any image dataset
- ✅ Trains Wemb module effectively
- ✅ Fast to start

**Result:** Wemb learns to embed diverse semantic vectors into images

### Strategy 2: Caption-Based Training

**Best for:** Production systems with caption data

```yaml
train:
  target: tools.dataset.datasetWithPrompts
  params:
    data_dir: /data/images
    data_list: data/train.csv
    caption_file: data/captions.csv  # Real captions
    resize: 256
```

**Advantages:**
- ✅ Content-specific watermarks
- ✅ Better semantic binding
- ✅ Ready for T2I deployment

**Requirements:**
- Need caption data for training images
- More preprocessing work

### Strategy 3: Hybrid Approach

Use generic prompts for training, then fine-tune with captions:

```bash
# Phase 1: Train with generic prompts
python train.py --config configs/SD14_SemanticLaWa.yaml --max_epochs 30

# Phase 2: Fine-tune with captions
python train.py --config configs/SD14_SemanticLaWa_captions.yaml \
    --checkpoint results/semantic_lawa_training/checkpoints/last.ckpt \
    --max_epochs 10
```

## Prompt Generation Details

### Default Prompt Pool

The system generates 100+ diverse prompts by combining:

**Templates:**
- "A photo", "An image", "A picture", "A photograph"
- "A colorful image", "A detailed photo", "A scenic view"
- "A beautiful picture", "A natural scene", etc.

**Contexts:**
- "with natural lighting", "with vivid colors", "with soft tones"
- "in daylight", "at sunset", "with high contrast"
- "captured professionally", "showing textures", etc.

**Example generated prompts:**
- "A photo with natural lighting"
- "A detailed photo in daylight"
- "A colorful image with high contrast"
- "A scenic view at sunset"

### How Prompts Are Assigned

```python
# Each image path gets a consistent prompt within an epoch
idx = hash(image_path) % len(prompt_pool)
prompt = prompt_pool[idx]
```

This ensures:
- Same image gets same prompt in each epoch (for consistency)
- Different images get different prompts (for diversity)
- No need to store prompt assignments

## Text Fusion in Training

**Question:** "How does text get fused during training?"

**Answer:** The fusion happens in the semantic encoder + Wemb module:

```python
# Step 1: Text → Semantic Vector
prompt = "A beautiful landscape"
semantic_vector = clip_encoder(prompt)  # 512-dim vector

# Step 2: Encrypt for security
encrypted_vector = rotation_matrix @ semantic_vector

# Step 3: Fuse with image features
# Inside Wemb module:
semantic_spatial = projection_layer(encrypted_vector)  # Vector → Spatial map
combined = concat(semantic_spatial, vae_features)      # Fuse with image
perturbation = unet(combined)                          # Generate perturbation
watermarked = vae_features + perturbation * strength   # Add to image
```

**Key insight:** The semantic vector is **projected to spatial feature maps** at each decoder layer, then processed by U-Net to generate content-adaptive perturbations.

## Training Tips

### 1. Start with Lightweight U-Net

```yaml
use_lightweight_wemb: True  # Faster training
```

Benefits:
- 30-50% faster training
- 30-40% less memory
- Good for experimentation

### 2. Monitor Cosine Similarity

During training, watch the `train/cosine_sim` metric:

- **Early training**: 0.3-0.5 (learning)
- **Mid training**: 0.6-0.8 (improving)
- **Converged**: >0.85 (good watermarking)

### 3. Adjust Watermark Strength

If watermarks are too visible:
```yaml
watermark_addition_weight: 0.05  # Reduce from 0.1
```

If extraction accuracy is low:
```yaml
watermark_addition_weight: 0.15  # Increase from 0.1
```

### 4. Balance Loss Weights

```yaml
recon_loss_weight: 0.1      # Image quality
adversarial_loss_weight: 1.0  # Realism
semantic_loss_weight: 2.0    # Watermark accuracy ← Most important
perceptual_loss_weight: 1.0  # Perceptual quality
```

## Verification After Training

Test your trained model:

```python
# Load model
model = SemanticLaWa.load_from_checkpoint('checkpoint.ckpt')

# Generate watermarked image
prompt = "Test prompt"
semantic_vec = model.semantic_encoder(prompt, encrypt=True)
watermarked = model(latent, image, semantic_vec)

# Extract and verify
extracted = model.decoder(watermarked)
decrypted = model.semantic_encoder.rotation_matrix.decrypt(extracted)
is_authentic, similarity = model.semantic_encoder.verify(decrypted, prompt)

print(f"Similarity: {similarity:.4f}")  # Should be >0.85
print(f"Authentic: {is_authentic}")     # Should be True
```

## Common Questions

### Q1: Do I need real captions to train?

**No.** You can train effectively with generic prompts. The Wemb module learns to embed any semantic vector, not just caption-specific ones.

### Q2: Can I use different prompts for each image in each epoch?

**Yes.** Modify `datasetWithPrompts` to randomize prompts differently. However, the current hash-based approach provides good consistency while maintaining diversity.

### Q3: How do I add captions to my dataset?

Create a CSV file matching your data_list format:

```python
import pandas as pd

# Load image list
images = pd.read_csv('data/train_100k.csv')

# Add captions (example with random captions)
images['caption'] = ["A photo" for _ in range(len(images))]

# Save
images[['path', 'caption']].to_csv('data/train_captions.csv', index=False)
```

### Q4: Can I train with images first, then add text later?

**Yes.** That's exactly the recommended workflow:

1. Train Wemb with images + generic prompts (Phase 1)
2. Optionally fine-tune with captions (Phase 2)
3. Use with T2I generation (Phase 3)

## Expected Training Results

After 30-40 epochs with default settings:

| Metric | Target | Description |
|--------|--------|-------------|
| Cosine Similarity | >0.90 | Clean image watermark accuracy |
| PSNR | >40 dB | Image quality |
| SSIM | >0.98 | Structural similarity |
| Training Time | ~30 hrs | On single GPU (V100/A100) |

## Troubleshooting

### Issue: Low cosine similarity (<0.70)

**Solutions:**
1. Increase `semantic_loss_weight` to 3.0-4.0
2. Reduce learning rate to 0.00004
3. Train longer (50+ epochs)
4. Check if semantic encoder is frozen (should be)

### Issue: Watermarks too visible

**Solutions:**
1. Reduce `watermark_addition_weight` to 0.05
2. Increase `recon_loss_weight` to 0.2
3. Add more perceptual loss weight

### Issue: Out of memory

**Solutions:**
1. Set `use_lightweight_wemb: True`
2. Reduce batch size to 4
3. Use mixed precision training
4. Reduce image resolution to 256×256

## Next Steps

After successful training:

1. **Evaluate robustness**: Test against JPEG compression, blur, rotation
2. **Deploy for inference**: Use `inference_semantic.py` for T2I generation
3. **Fine-tune**: Optionally fine-tune with real captions
4. **Integrate**: Add to your T2I pipeline (Stable Diffusion, etc.)

## Summary

The semantic watermarking training workflow:

1. **Prepare data**: Images + prompts (generic or real captions)
2. **Train Wemb**: Learn to embed semantic vectors into images
3. **Verify**: Check cosine similarity and image quality
4. **Deploy**: Use for T2I generation with content-specific watermarks

Each image can have its own unique watermark based on its prompt, enabling:
- Content authentication
- Copyright protection
- Usage tracking
- Metadata embedding

The system is flexible: start with generic prompts, upgrade to captions later, and deploy for production T2I systems.

# Semantic Watermarking for LaWa

This document describes the semantic watermarking upgrade for LaWa, which replaces binary bit streams with high-dimensional semantic vectors for enhanced security and information density.

## Overview

The semantic watermarking system embeds watermarks based on the **meaning** of the text prompt rather than arbitrary binary codes. This approach provides:

- **Stronger semantic binding** between watermark and image content
- **Higher information density** - can embed rich metadata
- **Enhanced security** through cryptographic rotation matrix encryption
- **Unforgeable watermarks** that are deeply tied to the generation prompt

## Architecture

### 1. Semantic Embedding Module (`models/semanticEmbedding.py`)

**Key Components:**
- **SemanticEncoder**: Uses CLIP text encoder to convert prompts to high-dimensional vectors (512/768 dims)
- **RotationMatrix**: Implements orthogonal rotation matrix encryption/decryption
- **Prompt Augmentation**: Optionally enriches prompts with metadata (model version, user ID, timestamp)

**Usage:**
```python
from models.semanticEmbedding import SemanticEncoder

# Initialize encoder
encoder = SemanticEncoder(rotation_seed=42)

# Encode prompt
prompt = "A photo of an astronaut riding a horse on the moon"
semantic_vector = encoder(prompt, encrypt=True)  # Encrypted vector

# Verify watermark
extracted_vector = decoder(watermarked_image)
decrypted = encoder.rotation_matrix.decrypt(extracted_vector)
is_authentic, similarity = encoder.verify(decrypted, prompt)
```

### 2. U-Net WEmb Module (`models/unetWEmb.py`)

**Key Components:**
- **UNetWEmb**: Full U-Net architecture for generating content-adaptive perturbations
- **LightweightUNetWEmb**: Efficient version for faster training/inference
- **SemanticVectorProjection**: Projects semantic vectors to spatial feature maps

**Features:**
- Multi-scale feature extraction and fusion
- Skip connections for preserving low and high-level features
- Zero-initialized output for stable training
- Content-adaptive perturbation generation

**Architecture:**
```
Semantic Vector (512) → Projection → Spatial Features (H×W)
                                          ↓
                                    Concatenate with Feature Map
                                          ↓
                                    U-Net Encoder (3 layers)
                                          ↓
                                    Bottleneck
                                          ↓
                                    U-Net Decoder (3 layers + skip connections)
                                          ↓
                                    Perturbation Pattern (same size as input)
```

### 3. Semantic Decoder (`models/semanticDecoder.py`)

**Key Components:**
- **SemanticDecoder**: ResNet50-based extractor for semantic vectors
- **LightweightSemanticDecoder**: MobileNetV2-based for efficiency
- **MultiScaleSemanticDecoder**: Enhanced robustness through multi-scale features

### 4. Semantic LaWa Model (`models/semanticLaWa.py`)

The main watermarking model that integrates all components:

**Training Loss:**
```python
total_loss = (
    recon_weight * reconstruction_loss +
    adversarial_weight * adversarial_loss +
    semantic_weight * (1 - cosine_similarity) +
    perceptual_weight * lpips_loss
)
```

**Workflow:**
1. Encode prompt to semantic vector
2. Encrypt vector with rotation matrix
3. Generate image latent with Stable Diffusion
4. Inject semantic perturbations at multiple VAE decoder layers
5. Decode to watermarked image

## Installation

### Dependencies
```bash
pip install torch torchvision transformers einops
```

### Model Setup
1. Download Stable Diffusion v1.4 weights
2. Download VAE KL-f8 weights
3. (Optional) Train Semantic LaWa model

## Usage

### Training

```bash
python train.py \
    --config configs/SD14_SemanticLaWa.yaml \
    --batch_size 8 \
    --max_epochs 40 \
    --learning_rate 0.00006 \
    --output results/semantic_lawa_training
```

### Inference

```bash
python inference_semantic.py \
    --config_sd stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
    --ckpt_sd weights/stable-diffusion-v1/model.ckpt \
    --config_lawa configs/SD14_SemanticLaWa_inference.yaml \
    --ckpt_lawa weights/semantic_lawa/last.ckpt \
    --prompt "A white plate of food on a dining table" \
    --add_metadata \
    --model_version "v1.0" \
    --user_id "12345" \
    --outdir results/semantic_watermarking
```

### Watermark Verification

The inference script automatically extracts and verifies watermarks. You can also do it manually:

```python
from models.semanticEmbedding import SemanticEncoder
from models.semanticDecoder import SemanticDecoder
import torch
from PIL import Image
from torchvision import transforms

# Load models
encoder = SemanticEncoder(rotation_seed=42)
decoder = SemanticDecoder(semantic_dim=512)
decoder.load_state_dict(torch.load('path/to/decoder.pth'))

# Load and preprocess image
image = Image.open('watermarked_image.png')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# Extract watermark
with torch.no_grad():
    extracted = decoder(image_tensor)
    decrypted = encoder.rotation_matrix.decrypt(extracted)

# Verify
original_prompt = "A photo of an astronaut riding a horse"
is_authentic, similarity = encoder.verify(decrypted, original_prompt)
print(f"Authentic: {is_authentic}, Similarity: {similarity:.4f}")
```

## Technical Details

### Rotation Matrix Encryption

The rotation matrix provides cryptographic security:

1. **Generation**: Use QR decomposition of random matrix to create orthogonal rotation matrix
2. **Encryption**: `encrypted = semantic_vector @ rotation_matrix`
3. **Decryption**: `decrypted = encrypted @ rotation_matrix.T` (inverse = transpose for orthogonal matrices)
4. **Properties**:
   - Preserves vector norms
   - Invertible with perfect reconstruction
   - Computationally efficient
   - Secure without the private key

### Watermark Embedding Points

Watermarks are injected at 6 strategic points in the VAE decoder:

1. **Initial latent** (4 channels, lowest resolution)
2. **Middle features** (512 channels, after attention)
3. **Upsampling layer 3** (128 channels)
4. **Upsampling layer 2** (256 channels)
5. **Upsampling layer 1** (512 channels)
6. **Upsampling layer 0** (512 channels, highest resolution)

### Cosine Similarity Verification

Watermark authenticity is determined by cosine similarity:

```
similarity = (extracted · original) / (||extracted|| × ||original||)
```

- **Threshold**: Default 0.85
- **Authentic**: similarity ≥ threshold
- **Forged/Corrupted**: similarity < threshold

## Configuration

### Key Parameters in Config Files

```yaml
model:
  params:
    rotation_seed: 42  # Seed for rotation matrix (private key)
    use_lightweight_wemb: True  # Use lightweight U-Net for efficiency
    cosine_similarity_threshold: 0.85  # Verification threshold
    watermark_addition_weight: 0.1  # Watermark strength
    semantic_loss_weight: 2.0  # Weight for semantic loss
    
    semantic_encoder_config:
      params:
        model_name: openai/clip-vit-base-patch32
        embedding_dim: 512  # Semantic vector dimension
    
    decoder_config:
      params:
        semantic_dim: 512  # Must match encoder
```

## Performance Metrics

Expected performance (after training):

| Metric | Value |
|--------|-------|
| PSNR | >40 dB |
| SSIM | >0.98 |
| Cosine Similarity (clean) | >0.90 |
| Cosine Similarity (attacked) | >0.75 |
| Extraction Accuracy | >85% |

## Comparison with Binary LaWa

| Feature | Binary LaWa | Semantic LaWa |
|---------|-------------|---------------|
| Watermark Type | 48-bit binary | 512-dim semantic vector |
| Information Density | 48 bits | ~2048 bits (equivalent) |
| Security | Moderate | High (cryptographic) |
| Semantic Binding | Weak | Strong |
| Metadata Support | Limited | Rich |
| Forgery Resistance | Moderate | High |
| Prompt Integration | No | Yes |

## Attacks and Robustness

The system is tested against common attacks:

- **Geometric**: Rotation, scaling, cropping
- **Filtering**: Gaussian blur, median filter
- **Compression**: JPEG (quality 50-90)
- **Noise**: Gaussian, salt-and-pepper
- **Brightness/Contrast**: ±20% variation

Target robustness: >75% similarity after attacks

## Future Enhancements

Potential improvements:

1. **Multi-modal Encoding**: Include visual features alongside text
2. **Adaptive Strength**: Adjust watermark strength based on content
3. **Hierarchical Watermarks**: Embed multiple levels of information
4. **Zero-knowledge Verification**: Verify without revealing the key
5. **Differential Privacy**: Add privacy-preserving mechanisms

## Citation

If you use Semantic LaWa in your research, please cite:

```bibtex
@misc{semantic_lawa_2025,
    title={Semantic LaWa: Semantic Watermarking for Text-to-Image Generation},
    author={Your Name},
    year={2025},
    note={Upgraded version of LaWa with semantic vectors}
}
```

## References

1. **LatentSeal**: Semantic watermarking with rotation matrix encryption
2. **SWIFT**: Semantic watermark integration for text-to-image
3. **MetaSeal**: Metadata-based watermarking with unforgeable verification
4. **Embedding Guide**: UNet++ for multi-scale feature fusion
5. **Original LaWa**: Using Latent Space for In-Generation Image Watermarking (ECCV 2024)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases in `tests/test_semantic_modules.py`

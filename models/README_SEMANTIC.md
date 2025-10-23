# Semantic Watermarking Modules

This directory contains the semantic watermarking implementation modules.

## Module Overview

### semanticEmbedding.py

**Purpose:** Text-to-vector encoding and cryptographic encryption

**Classes:**
- `RotationMatrix`: Implements orthogonal rotation matrix for encryption/decryption
- `SemanticEncoder`: Converts text prompts to semantic vectors using CLIP

**Key Features:**
- CLIP text encoder integration (with fallback mode)
- Rotation matrix encryption for security
- Prompt augmentation with metadata
- Cosine similarity verification

**Usage Example:**
```python
from models.semanticEmbedding import SemanticEncoder

encoder = SemanticEncoder(rotation_seed=42)
prompt = "A photo of an astronaut"
semantic_vector = encoder(prompt, encrypt=True)
```

### unetWEmb.py

**Purpose:** U-Net based watermark embedding modules

**Classes:**
- `UNetWEmb`: Full U-Net for perturbation generation
- `LightweightUNetWEmb`: Efficient version for faster training
- `MultiScaleUNetWEmb`: Multi-layer watermark injection
- `SemanticVectorProjection`: Projects semantic vectors to spatial features

**Key Features:**
- Content-adaptive perturbation generation
- Multi-scale feature processing
- Skip connections for detail preservation
- Zero-initialized output

**Usage Example:**
```python
from models.unetWEmb import LightweightUNetWEmb

wemb = LightweightUNetWEmb(
    semantic_dim=512,
    feature_channels=256,
    base_channels=32
)
perturbation = wemb(semantic_vector, feature_map)
```

### semanticDecoder.py

**Purpose:** Extract semantic vectors from watermarked images

**Classes:**
- `SemanticDecoder`: ResNet50-based extractor
- `LightweightSemanticDecoder`: MobileNetV2-based for efficiency
- `MultiScaleSemanticDecoder`: Enhanced robustness

**Key Features:**
- Pre-trained backbone (ResNet50/MobileNetV2)
- Projection to semantic space
- Optional multi-scale features

**Usage Example:**
```python
from models.semanticDecoder import SemanticDecoder

decoder = SemanticDecoder(semantic_dim=512)
extracted_semantic = decoder(watermarked_image)
```

### semanticLaWa.py

**Purpose:** Main semantic watermarking model

**Class:**
- `SemanticLaWa`: Integrates all components for end-to-end watermarking

**Key Features:**
- 6-point watermark injection in VAE decoder
- Cosine similarity loss
- Support for metadata
- Compatible with Stable Diffusion

**Usage Example:**
```python
from models.semanticLaWa import SemanticLaWa
from omegaconf import OmegaConf

config = OmegaConf.load('configs/SD14_SemanticLaWa.yaml')
model = SemanticLaWa(**config.model.params)
```

## Module Dependencies

```
semanticLaWa.py
├── semanticEmbedding.py (text encoding + encryption)
├── unetWEmb.py (perturbation generation)
├── semanticDecoder.py (watermark extraction)
└── modifiedAEDecoder.py (discriminator)
```

## Integration with Original LaWa

### Shared Components
- VAE encoder/decoder (from `first_stage_config`)
- Discriminator (`Discriminator1`)
- Training loop structure
- Augmentation/noise modules

### New Components
- Semantic encoder (replaces binary message generation)
- U-Net WEmb (replaces linear+conv watermark layers)
- Semantic decoder (replaces binary message decoder)
- Rotation matrix encryption (new security layer)

## Configuration

### Training Config Example

```yaml
model:
  target: models.semanticLaWa.SemanticLaWa
  params:
    rotation_seed: 42  # Private key for encryption
    use_lightweight_wemb: True  # Use efficient U-Net
    
    semantic_encoder_config:
      target: models.semanticEmbedding.SemanticEncoder
      params:
        embedding_dim: 512
    
    decoder_config:
      target: models.semanticDecoder.SemanticDecoder
      params:
        semantic_dim: 512
    
    semantic_loss_weight: 2.0  # Weight for cosine similarity loss
```

## Loss Functions

### Semantic Loss (Main Watermark Loss)

```python
# Extract and decrypt
pred_semantic = decoder(watermarked_image)
pred_decrypted = rotation_matrix.decrypt(pred_semantic)
target_decrypted = rotation_matrix.decrypt(target_semantic)

# Cosine similarity loss
similarity = F.cosine_similarity(pred_decrypted, target_decrypted, dim=-1)
semantic_loss = 1 - similarity.mean()
```

### Total Training Loss

```python
total_loss = (
    recon_weight * reconstruction_loss +      # Image quality
    adversarial_weight * discriminator_loss + # Realism
    semantic_weight * semantic_loss +         # Watermark fidelity
    perceptual_weight * lpips_loss           # Perceptual quality
)
```

## Performance Considerations

### Memory Usage

| Module | Parameters | Memory (approx) |
|--------|-----------|----------------|
| SemanticEncoder | ~150M (CLIP) | ~600 MB |
| UNetWEmb (full) | ~15M | ~60 MB |
| LightweightUNetWEmb | ~3M | ~12 MB |
| SemanticDecoder | ~25M | ~100 MB |

**Recommendation:** Use `use_lightweight_wemb: True` for training on limited GPU memory.

### Training Speed

- **Binary LaWa**: 1x baseline
- **Semantic LaWa (lightweight)**: ~1.2x slower
- **Semantic LaWa (full)**: ~1.5x slower

Trade-off between speed and watermark robustness.

## Troubleshooting

### Issue: CLIP Model Not Loading

**Symptom:** "Could not load CLIP model" warning

**Solution:** The system automatically uses fallback mode with hash-based encoding. For production:
```bash
pip install transformers protobuf
```

### Issue: CUDA Out of Memory

**Solution:** Use lightweight modules:
```yaml
use_lightweight_wemb: True
```

Or reduce batch size:
```yaml
data:
  params:
    batch_size: 4  # Reduce from 8
```

### Issue: Low Cosine Similarity During Training

**Possible Causes:**
1. Learning rate too high
2. Semantic loss weight too low
3. Insufficient training epochs

**Solutions:**
```yaml
learning_rate: 0.00004  # Reduce
semantic_loss_weight: 3.0  # Increase
```

## Testing

Run unit tests for all modules:
```bash
python tests/test_semantic_modules.py
```

Expected output:
```
Running Semantic Watermarking Module Tests
============================================================
✓ Rotation matrix test passed
✓ Semantic encoder test passed
✓ U-Net WEmb test passed
✓ Semantic decoder test passed
✓ Integration test passed
============================================================
Tests Results: 5 passed, 0 failed
```

## Further Reading

- [SEMANTIC_WATERMARKING.md](../SEMANTIC_WATERMARKING.md) - Complete technical documentation
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migration from binary LaWa
- [examples/semantic_watermarking_demo.py](../examples/semantic_watermarking_demo.py) - Working examples

## Module Versions

- **semanticEmbedding.py**: v1.0 - Initial implementation
- **unetWEmb.py**: v1.0 - Initial implementation
- **semanticDecoder.py**: v1.0 - Initial implementation
- **semanticLaWa.py**: v1.0 - Initial implementation

## License

Same license as the main LaWa repository.

## Contributors

Implemented based on issues #1, #2, and #3 specifications.

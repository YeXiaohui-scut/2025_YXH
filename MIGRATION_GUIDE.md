# Migration Guide: From Binary LaWa to Semantic LaWa

This guide helps you migrate from the original binary watermarking system to the new semantic watermarking system.

## Overview of Changes

| Component | Binary LaWa | Semantic LaWa |
|-----------|-------------|---------------|
| Input | 48-bit binary message | Text prompt + optional metadata |
| Embedding Module | Linear + Conv layers | U-Net perturbation generator |
| Decoder | ResNet50 → 48 bits | ResNet50 → 512-dim vector |
| Verification | Bit accuracy (XOR) | Cosine similarity |
| Security | None | Rotation matrix encryption |

## Step-by-Step Migration

### 1. Update Dependencies

No new dependencies are required beyond the original LaWa requirements. The semantic modules use the same base libraries (torch, torchvision).

Optional: Install `transformers` for CLIP text encoder (can work without it using fallback mode).

```bash
pip install transformers  # Optional
```

### 2. Replace Model Configuration

**Old config (`configs/SD14_LaWa.yaml`):**
```yaml
model:
  target: models.modifiedAEDecoder.LaWa
  params:
    decoder_config:
      target: models.messageDecoder.MessageDecoder
      params: 
        message_len: 48
```

**New config (`configs/SD14_SemanticLaWa.yaml`):**
```yaml
model:
  target: models.semanticLaWa.SemanticLaWa
  params:
    rotation_seed: 42
    use_lightweight_wemb: True
    cosine_similarity_threshold: 0.85
    
    semantic_encoder_config:
      target: models.semanticEmbedding.SemanticEncoder
      params:
        embedding_dim: 512
    
    decoder_config:
      target: models.semanticDecoder.SemanticDecoder
      params: 
        semantic_dim: 512
    
    semantic_loss_weight: 2.0  # New loss term
```

### 3. Update Training Code

**Old training:**
```python
# Binary message generation
message = torch.randint(0, 2, (batch_size, 48)).float() * 2 - 1

# Forward pass
watermarked_image = model(latent, image, message)

# Loss calculation
pred = decoder(watermarked_image)
loss = bce_loss(pred, 0.5 * (message + 1))
```

**New training:**
```python
# Prompt-based semantic vector (or random for training)
prompts = batch['prompt']  # From dataset
semantic_vector = model.semantic_encoder(prompts, encrypt=True)

# Forward pass (same interface)
watermarked_image = model(latent, image, semantic_vector)

# Loss calculation with cosine similarity
pred_semantic = decoder(watermarked_image)
decrypted = model.semantic_encoder.rotation_matrix.decrypt(pred_semantic)
target = model.semantic_encoder.rotation_matrix.decrypt(semantic_vector)
similarity = F.cosine_similarity(decrypted, target, dim=-1)
loss = 1 - similarity.mean()
```

### 4. Update Dataset

The semantic system needs text prompts. Two options:

#### Option A: Add Prompts to Existing Dataset

Create a dataset class that includes prompts:

```python
class DatasetWithPrompts(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list, resize=256):
        # Load your existing data
        self.images = ...
        # Add prompts (can be random/generic for training)
        self.prompts = self.generate_prompts()
    
    def generate_prompts(self):
        # Simple strategy: generic prompts
        generic_prompts = [
            "A photo",
            "An image",
            "A picture",
            # ... more prompts
        ]
        return [random.choice(generic_prompts) for _ in range(len(self.images))]
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'prompt': self.prompts[idx]
        }
```

#### Option B: Use Random Semantic Vectors (Simpler)

For training without real prompts, modify `get_input()` to generate random vectors:

```python
def get_input(self, batch, bs=None):
    # ... existing code ...
    
    # Generate random semantic vectors (for training without prompts)
    semantic_vector = torch.randn(bs, self.semantic_dim, device=image.device)
    semantic_vector = self.semantic_encoder.rotation_matrix.encrypt(semantic_vector)
    
    return [x, semantic_vector, image, image_rec, None]
```

### 5. Update Inference Code

**Old inference:**
```python
# Binary message
message = '110111001110110001000000011101000110011100110101'
message_tensor = torch.tensor([int(b) for b in message]).float() * 2 - 1

# Generate
watermarked = model(latent, image, message_tensor)

# Verify
extracted = decoder(watermarked)
bit_accuracy = (extracted > 0) == (message_tensor > 0)
```

**New inference:**
```python
# Prompt-based watermark
prompt = "A white plate of food on a dining table"
metadata = {'model_version': 'v1.0', 'user_id': '12345'}

# Generate
semantic_vector = model.semantic_encoder(prompt, encrypt=True, metadata=metadata)
watermarked = model(latent, image, semantic_vector)

# Verify
extracted = decoder(watermarked)
decrypted = model.semantic_encoder.rotation_matrix.decrypt(extracted)
is_authentic, similarity = model.semantic_encoder.verify(
    decrypted, prompt, threshold=0.85, metadata=metadata
)
print(f"Authentic: {is_authentic}, Similarity: {similarity:.4f}")
```

### 6. Weight Conversion (Optional)

If you have trained binary LaWa weights, you can partially reuse them:

```python
# Load binary LaWa checkpoint
binary_checkpoint = torch.load('binary_lawa.ckpt')

# Create semantic LaWa model
semantic_model = SemanticLaWa(...)

# Transfer shared weights (VAE, discriminator)
semantic_model.ae.load_state_dict(binary_checkpoint['ae'], strict=False)
semantic_model.discriminator.load_state_dict(binary_checkpoint['discriminator'])

# WEmb modules need retraining (different architecture)
# Decoder needs retraining (different output dimension)
```

### 7. Adjust Hyperparameters

Recommended starting values for semantic watermarking:

```yaml
learning_rate: 0.00006  # Slightly lower than binary
semantic_loss_weight: 2.0  # Main watermark loss
watermark_addition_weight: 0.1  # Injection strength
cosine_similarity_threshold: 0.85  # Verification threshold
```

### 8. Testing and Validation

After migration, validate that:

1. **Training converges**: Monitor cosine similarity (should reach >0.90 on clean images)
2. **Quality preserved**: PSNR >40 dB, SSIM >0.98
3. **Robustness maintained**: Similarity >0.75 after attacks
4. **Verification works**: High similarity for authentic prompts, low for wrong prompts

```python
# Validation script
python tests/test_semantic_modules.py  # Unit tests
python examples/semantic_watermarking_demo.py  # Integration demo
```

## Common Migration Issues

### Issue 1: "CLIP model not loading"

**Solution:** Set `embedding_dim` in config and use fallback mode:
```yaml
semantic_encoder_config:
  params:
    embedding_dim: 512  # Fallback dimension
```

Or install transformers:
```bash
pip install transformers protobuf
```

### Issue 2: "Low cosine similarity during training"

**Causes:**
- Learning rate too high
- Semantic loss weight too low
- Insufficient training

**Solutions:**
```yaml
learning_rate: 0.00004  # Reduce
semantic_loss_weight: 3.0  # Increase
```

### Issue 3: "Watermark too visible"

**Solution:** Reduce injection strength:
```yaml
watermark_addition_weight: 0.05  # Reduce from 0.1
```

### Issue 4: "Memory errors with full U-Net"

**Solution:** Use lightweight version:
```yaml
use_lightweight_wemb: True
```

## Compatibility Notes

### Backward Compatibility

- Old binary LaWa models are **not directly compatible** with semantic verification
- You can run both systems side-by-side using different configs
- Original inference scripts (`inference_AIGC.py`) still work for binary watermarks

### Forward Migration Path

1. Start with lightweight semantic LaWa for faster iteration
2. Train with random semantic vectors if you don't have prompts
3. Gradually increase model capacity (full U-Net) as needed
4. Fine-tune with real prompts for production use

## Performance Comparison

After migration, expect:

| Metric | Binary LaWa | Semantic LaWa | Change |
|--------|-------------|---------------|--------|
| Training Time | 1x | 1.2-1.5x | +20-50% |
| Inference Time | 1x | 1.1-1.3x | +10-30% |
| Memory Usage | 1x | 1.3-1.6x | +30-60% |
| Watermark Capacity | 48 bits | ~2048 bits | +4166% |
| Security | Low | High | ++ |

## Migration Checklist

- [ ] Update model configuration
- [ ] Install optional dependencies (transformers)
- [ ] Modify dataset to include prompts (or use random vectors)
- [ ] Update training loop for semantic loss
- [ ] Adjust hyperparameters
- [ ] Update inference scripts
- [ ] Test with validation data
- [ ] Verify quality metrics (PSNR, SSIM)
- [ ] Test robustness against attacks
- [ ] Update documentation and scripts

## Getting Help

If you encounter issues during migration:

1. Check [SEMANTIC_WATERMARKING.md](SEMANTIC_WATERMARKING.md) for detailed documentation
2. Review examples in `examples/semantic_watermarking_demo.py`
3. Run unit tests: `python tests/test_semantic_modules.py`
4. Check the three GitHub issues (#1, #2, #3) for technical details
5. Open a new issue with your specific problem

## Summary

The migration from binary to semantic watermarking involves:

1. **Model change**: `modifiedAEDecoder.LaWa` → `semanticLaWa.SemanticLaWa`
2. **Input change**: Binary bits → Text prompts + semantic vectors
3. **Architecture change**: Linear layers → U-Net perturbation generators
4. **Verification change**: Bit accuracy → Cosine similarity
5. **Security addition**: Rotation matrix encryption

The migration provides significant benefits in security, capacity, and semantic binding, making it worthwhile for production systems.

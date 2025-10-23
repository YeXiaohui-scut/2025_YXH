# Examples

This directory contains example scripts demonstrating the semantic watermarking system.

## Available Examples

### semantic_watermarking_demo.py

Comprehensive demo showing all components of the semantic watermarking system:

1. **Rotation Matrix Encryption** - How cryptographic encryption works
2. **Semantic Text Encoding** - Converting text to semantic vectors
3. **Watermark Embedding** - Generating content-adaptive perturbations with U-Net
4. **Watermark Extraction** - Extracting and verifying watermarks
5. **End-to-End Pipeline** - Complete watermarking workflow

**Run the demo:**
```bash
python examples/semantic_watermarking_demo.py
```

## Expected Output

The demo will print information about each component:

```
============================================================
SEMANTIC WATERMARKING SYSTEM DEMO
============================================================

============================================================
Demo 1: Rotation Matrix Encryption
============================================================
Original vector norm: 22.6274
Encrypted vector norm: 22.6274
Decrypted vector norm: 22.6274
Reconstruction error: 3.04e-06

✓ Rotation matrix preserves vector norms and enables perfect reconstruction

...
```

## Understanding the Output

- **Rotation Matrix Demo**: Shows that encryption preserves vector properties and enables perfect decryption
- **Semantic Encoding**: Demonstrates how text prompts are converted to vectors
- **Watermark Embedding**: Shows perturbation generation statistics
- **Extraction & Verification**: Displays cosine similarity scores for authentic and fake prompts
- **End-to-End**: Complete pipeline from prompt to verification

## Customization

You can modify the demo to:
- Try different prompts
- Adjust semantic dimensions
- Change watermark strength
- Test with different feature map sizes
- Experiment with metadata

## Next Steps

After running the demo:

1. Read the [full documentation](../SEMANTIC_WATERMARKING.md)
2. Try the inference script with real Stable Diffusion models
3. Train your own semantic watermarking model
4. Run unit tests in `tests/test_semantic_modules.py`

## Troubleshooting

**CLIP model not loading**: The demo uses a fallback mode with hash-based deterministic encoding. For production use, ensure transformers and CLIP are properly installed.

**Low similarity scores in demo**: The demo uses random images without actual watermark training. Real trained models achieve 85%+ similarity.

**Import errors**: Make sure you're running from the repository root directory.

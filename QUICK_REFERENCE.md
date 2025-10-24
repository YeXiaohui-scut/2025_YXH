# Semantic Watermarking: Quick Reference

## Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                             │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Data Loading & Prompt Assignment
┌──────────────┐    ┌──────────────────────────┐
│   Dataset    │───→│  Assign Prompts          │
│  (Images)    │    │  - Generic (random pool) │
│              │    │  - OR Captions (if avail)│
└──────────────┘    └──────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │ Each Image + Prompt     │
                    │ img1.jpg → "A photo"    │
                    │ img2.jpg → "A landscape"│
                    └─────────────────────────┘

Phase 2: Semantic Encoding & Encryption
                              ↓
                    ┌─────────────────────────┐
                    │   CLIP Text Encoder     │
                    │   (or hash fallback)    │
                    │ "A photo" → [512 dims]  │
                    └─────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │  Rotation Matrix        │
                    │  Encryption (secure)    │
                    │  vector @ R_matrix      │
                    └─────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │ Encrypted Semantic Vec  │
                    │    [512 dimensions]     │
                    └─────────────────────────┘

Phase 3: Watermark Embedding (6 injection points)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VAE DECODER (Modified)                        │
│                                                                  │
│  ┌──────────┐     ┌──────────────┐      ┌────────────────┐    │
│  │  Latent  │────→│ WEmb Module  │─────→│ + Perturbation │    │
│  │  (4 ch)  │     │  (U-Net 0)   │      │   * strength   │    │
│  └──────────┘     └──────────────┘      └────────────────┘    │
│       ↓                   ↑                       ↓             │
│  ┌──────────┐    Semantic Vector        ┌────────────────┐    │
│  │  Conv In │    (broadcast to spatial) │  Conv Layers   │    │
│  └──────────┘                            └────────────────┘    │
│       ↓                                           ↓             │
│  ┌──────────┐     ┌──────────────┐      ┌────────────────┐    │
│  │  Middle  │────→│ WEmb Module  │─────→│ + Perturbation │    │
│  │ (512 ch) │     │  (U-Net 1)   │      │                │    │
│  └──────────┘     └──────────────┘      └────────────────┘    │
│       ↓                   ↑                       ↓             │
│  ┌──────────┐    Semantic Vector        ┌────────────────┐    │
│  │ Upsample │    (fused with features)  │  Upsample 3    │    │
│  │ Layer 3  │                            │   (128 ch)     │    │
│  └──────────┘                            └────────────────┘    │
│       ↓                                           ↓             │
│   ... (4 more injection points) ...                            │
│       ↓                                           ↓             │
│  ┌──────────┐                           ┌────────────────┐    │
│  │  Output  │                           │  Watermarked   │    │
│  │  Conv    │                           │     Image      │    │
│  └──────────┘                           └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Phase 4: Watermark Extraction & Verification
                              ↓
                    ┌─────────────────────────┐
                    │  Semantic Decoder       │
                    │  (ResNet50)             │
                    │  Watermarked Image →    │
                    │  Extracted Vector       │
                    └─────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │  Rotation Matrix        │
                    │  Decryption             │
                    │  vector @ R_matrix^T    │
                    └─────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
                    │  Cosine Similarity      │
                    │  Compare with original  │
                    │  similarity > 0.85?     │
                    └─────────────────────────┘

Phase 5: Loss Computation
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LOSS FUNCTION                               │
│                                                                  │
│  Total Loss = Recon_Loss + Adversarial_Loss + Semantic_Loss     │
│                                                                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────┐ │
│  │ Reconstruction  │ + │  Adversarial    │ + │  Semantic    │ │
│  │ MSE(original,   │   │  Discriminator  │   │  1 - Cosine  │ │
│  │  watermarked)   │   │  on watermarked │   │  Similarity  │ │
│  │  × 0.1          │   │  × 1.0          │   │  × 2.0       │ │
│  └─────────────────┘   └─────────────────┘   └──────────────┘ │
│        ↓                       ↓                      ↓         │
│   Image Quality            Realism              Watermark      │
│   Preserved              Maintained             Accuracy       │
└─────────────────────────────────────────────────────────────────┘
```

## Text Fusion in WEmb Module

```
┌─────────────────────────────────────────────────────────────────┐
│              HOW TEXT IS FUSED WITH IMAGE                        │
└─────────────────────────────────────────────────────────────────┘

Input: Semantic Vector (512-dim) + Feature Map (C×H×W)

Step 1: Vector → Spatial Projection
┌───────────────┐
│  [512 dims]   │ ───→ Linear Layer ───→ ┌────────────────┐
│ Semantic Vec  │                         │  [64×8×8]      │
└───────────────┘                         │ Spatial Feature│
                                          └────────────────┘
                                                  ↓
                                          Upsample to H×W
                                                  ↓
                                          ┌────────────────┐
                                          │  [64×H×W]      │
                                          │ Spatial Feature│
                                          └────────────────┘

Step 2: Concatenate with Image Features
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  [64×H×W]      │  +  │  [C×H×W]       │  =  │ [(64+C)×H×W]   │
│ Semantic       │     │ Image Features │     │ Combined       │
│ Spatial        │     │ from VAE       │     │ Features       │
└────────────────┘     └────────────────┘     └────────────────┘

Step 3: U-Net Processing (Content-Adaptive)
┌────────────────────────────────────────────────────────────────┐
│                        U-Net WEmb                               │
│                                                                 │
│  ┌─────────────┐                                               │
│  │  Combined   │                                               │
│  │  Features   │                                               │
│  └─────────────┘                                               │
│         ↓                                                       │
│  ┌─────────────┐   Encoder (3 layers with skip connections)   │
│  │   Conv +    │   ┌─────┐ ┌─────┐ ┌─────┐                   │
│  │  Downsample │───│Skip1│─│Skip2│─│Skip3│                   │
│  └─────────────┘   └─────┘ └─────┘ └─────┘                   │
│         ↓              ↓       ↓       ↓                       │
│  ┌─────────────┐                                               │
│  │ Bottleneck  │   Decoder (3 layers with skip connections)   │
│  └─────────────┘      ↓       ↓       ↓                       │
│         ↓         ┌─────┐ ┌─────┐ ┌─────┐                    │
│  ┌─────────────┐ │     │ │     │ │     │                    │
│  │   Conv +    │─┴─────┴─┴─────┴─┴─────┴                    │
│  │  Upsample   │                                               │
│  └─────────────┘                                               │
│         ↓                                                       │
│  ┌─────────────┐                                               │
│  │ Perturbation│ Zero-initialized output                       │
│  │  [C×H×W]    │ (same size as input feature map)            │
│  └─────────────┘                                               │
└────────────────────────────────────────────────────────────────┘

Step 4: Add Perturbation to Features
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Original      │  +  │ Perturbation   │  =  │  Watermarked   │
│  Features      │     │ × strength     │     │  Features      │
│  [C×H×W]       │     │ [C×H×W]        │     │  [C×H×W]       │
└────────────────┘     └────────────────┘     └────────────────┘
                                                       ↓
                                              Continue VAE decode
                                                       ↓
                                              Watermarked Image

Key Points:
- Semantic vector is projected to EVERY decoder layer
- Each layer has its own U-Net WEmb module
- Perturbations are content-adaptive (vary with image content)
- Zero initialization prevents training instability
```

## Training Strategies Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                  STRATEGY 1: Generic Prompts                     │
├─────────────────────────────────────────────────────────────────┤
│ Data Needed:   Only images (no captions)                        │
│ Prompt Source: Auto-generated pool (100+ prompts)               │
│ Assignment:    hash(image_path) % pool_size                     │
│ Training Time: ~30 hours (40 epochs)                            │
│ Best For:      Initial training, no caption data                │
│ Result:        Wemb learns to embed ANY semantic vector         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STRATEGY 2: Caption-Based                       │
├─────────────────────────────────────────────────────────────────┤
│ Data Needed:   Images + captions.csv                            │
│ Prompt Source: Real image captions                              │
│ Assignment:    Load from caption file                           │
│ Training Time: ~30 hours (40 epochs)                            │
│ Best For:      Production, content-specific watermarks          │
│ Result:        Strong semantic binding to content               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STRATEGY 3: Hybrid                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1:       Train with generic prompts (30 epochs)           │
│ Phase 2:       Fine-tune with captions (10 epochs)              │
│ Training Time: ~35 hours total                                  │
│ Best For:      Gradual upgrade, limited caption data            │
│ Result:        Best of both worlds                              │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Command Reference

```bash
# Test semantic encoding
python examples/train_semantic_example.py --mode test_encoding

# View prompt diversity
python examples/train_semantic_example.py --mode demo_prompts

# Train with sample data (quick demo)
python examples/train_semantic_example.py \
    --mode train \
    --use_sample_data \
    --max_epochs 2

# Train with generic prompts (recommended start)
python train.py \
    --config configs/SD14_SemanticLaWa.yaml \
    --batch_size 8 \
    --max_epochs 40 \
    --learning_rate 0.00006 \
    --output results/semantic_training

# Train with captions (if available)
# First update config to set caption_file, then:
python train.py \
    --config configs/SD14_SemanticLaWa.yaml \
    --batch_size 8 \
    --max_epochs 40 \
    --output results/semantic_training_captions
```

## Metrics to Monitor

```
Training Metrics:
  train/cosine_sim      Target: >0.85  (Watermark accuracy)
  train/accuracy        Target: >0.85  (Pass rate at threshold)
  train/psnr            Target: >40 dB (Image quality)
  train/emb_loss        Should decrease steadily
  
Validation Metrics:
  val/cosine_sim        Target: >0.85
  val/psnr              Target: >40 dB

Progress Timeline:
  Epochs 1-10:   cosine_sim: 0.3 → 0.6
  Epochs 10-20:  cosine_sim: 0.6 → 0.8
  Epochs 20-30:  cosine_sim: >0.85 (converged)
  Epochs 30+:    Fine-tuning for robustness
```

## Troubleshooting Quick Reference

```
Problem: Low cosine similarity (<0.70)
Solution: 
  - Increase semantic_loss_weight to 3.0
  - Reduce learning rate to 0.00004
  - Train longer (50+ epochs)

Problem: Watermarks too visible
Solution:
  - Reduce watermark_addition_weight to 0.05
  - Increase recon_loss_weight to 0.2

Problem: Out of memory
Solution:
  - Set use_lightweight_wemb: True
  - Reduce batch_size to 4
  - Reduce image size to 256x256

Problem: CLIP not loading
Solution:
  - pip install transformers
  - OR use fallback mode (automatic)
```

## File Structure

```
Repository Root/
├── configs/
│   └── SD14_SemanticLaWa.yaml        # Training config
├── models/
│   ├── semanticLaWa.py               # Main model
│   ├── semanticEmbedding.py          # Text encoder
│   ├── semanticDecoder.py            # Watermark extractor
│   └── unetWEmb.py                   # Perturbation generator
├── tools/
│   └── dataset.py                     # Dataset with prompts
├── examples/
│   ├── train_semantic_example.py     # Training examples
│   └── README_TRAINING.md            # This file
├── TRAINING_GUIDE.md                 # Full guide (400+ lines)
├── SEMANTIC_WATERMARKING.md          # Architecture docs
└── train.py                          # Main training script
```

## Summary

**The Answer to "How to train Wemb with text?"**

1. **Images load** → Each gets a prompt (generic or caption)
2. **Prompt encodes** → 512-dim semantic vector (CLIP or hash)
3. **Vector encrypts** → Rotation matrix for security
4. **Vector projects** → Spatial feature maps at each layer
5. **Fusion happens** → Concatenate with VAE features
6. **U-Net processes** → Generates content-adaptive perturbations
7. **Perturbations add** → Inject into 6 decoder layers
8. **Training optimizes** → Maximize cosine similarity + preserve quality

**Result:** Each image can have a unique, content-related watermark based on its prompt!

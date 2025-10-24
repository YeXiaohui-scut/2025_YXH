# 2025_YXH

## 语义水印 - 基于文本提示的内容相关水印 / Semantic Watermarking for Text-to-Image

This repository implements semantic watermarking for LaWa (Latent Watermarking), enabling **content-related watermarks** where each generated image has a unique watermark derived from its text prompt.

### 🚀 开始使用 / Quick Start

**新用户从这里开始 / New users start here:**

👉 **[开始使用.md (5分钟快速上手)](开始使用.md)** - 最快的入门方式 / Fastest way to get started

或者选择你的路径 / Or choose your path:

1. **想直接开始训练?** → [开始使用.md](开始使用.md)
2. **想理解完整工作流程?** → [训练指南_中文.md](训练指南_中文.md) 或 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
3. **想看可视化流程图?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. **从二进制LaWa迁移?** → [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

### 📖 完整文档 / Complete Documentation

#### 中文文档 / Chinese Docs
| 文档 | 用途 | 时长 |
|------|------|------|
| [开始使用.md](开始使用.md) | 5分钟快速上手 | ⚡ 快速 |
| [训练指南_中文.md](训练指南_中文.md) | 完整训练工作流程 | 📚 详细 |

#### English Docs
| Document | Purpose | Length |
|----------|---------|--------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Complete training workflow | Comprehensive |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Visual diagrams, commands | Quick scan |
| [examples/README_TRAINING.md](examples/README_TRAINING.md) | Training examples | Tutorial |
| [SEMANTIC_WATERMARKING.md](SEMANTIC_WATERMARKING.md) | Architecture & tech | Reference |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Binary → Semantic | For upgrades |

### 🎯 Key Features

- **Content-Related Watermarks**: Each image gets a unique watermark based on its text prompt
- **High Security**: Cryptographic rotation matrix encryption
- **High Capacity**: 512-dimensional semantic vectors (vs 48-bit binary)
- **Flexible Training**: Works with or without image captions
- **Multiple Strategies**: Generic prompts, captions, or hybrid approach

### 💡 How It Works (In Brief)

```
Text Prompt → Semantic Vector → Encryption → Spatial Projection
                                                    ↓
Image → VAE Encoder → Latent → [ Inject Watermark ] → VAE Decoder → Watermarked Image
                                      ↑
                            U-Net WEmb (6 layers)
```

Each text prompt creates a unique 512-dimensional semantic vector that is:
1. **Encrypted** with a rotation matrix (cryptographic security)
2. **Projected** to spatial feature maps at each decoder layer
3. **Fused** with image features using U-Net modules
4. **Injected** as content-adaptive perturbations

### 🔧 Training Workflow

**Three strategies available:**

#### 1. Generic Prompts (Recommended to Start)
```bash
python train.py --config configs/SD14_SemanticLaWa.yaml
```
- No caption data needed
- Auto-generates 100+ diverse prompts
- Wemb learns to embed any semantic vector

#### 2. Caption-Based Training
```bash
# First, add caption_file to config
python train.py --config configs/SD14_SemanticLaWa.yaml
```
- Uses real image captions
- Content-specific watermarks
- Better semantic binding

#### 3. Hybrid Approach
```bash
# Phase 1: Train with generic prompts
python train.py --max_epochs 30 --output results/phase1

# Phase 2: Fine-tune with captions
python train.py --checkpoint results/phase1/last.ckpt --max_epochs 10
```

### 📊 Expected Results

After 30-40 epochs:
- **Cosine Similarity**: >0.85 (watermark accuracy)
- **PSNR**: >40 dB (image quality)
- **SSIM**: >0.98 (structural similarity)
- Each image has unique watermark based on its prompt

### 🎓 Understanding Text Fusion

**Question**: *"How does text get integrated during training?"*

**Answer**: The text prompt is encoded to a semantic vector, then:

1. **Projection**: Vector → Spatial feature maps (at each decoder layer)
2. **Concatenation**: Semantic features + VAE features
3. **Processing**: U-Net generates content-adaptive perturbations
4. **Injection**: Perturbations added to 6 strategic decoder layers

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for visual diagrams.

### 🧪 Testing & Examples

```bash
# Test semantic encoding
python examples/train_semantic_example.py --mode test_encoding

# View prompt diversity
python examples/train_semantic_example.py --mode demo_prompts

# Quick training demo with sample data
python examples/train_semantic_example.py --mode train --use_sample_data
```

### 📁 Repository Structure

```
├── configs/
│   └── SD14_SemanticLaWa.yaml           # Training configuration
├── models/
│   ├── semanticLaWa.py                  # Main watermarking model
│   ├── semanticEmbedding.py             # Text encoder + encryption
│   ├── semanticDecoder.py               # Watermark extractor
│   └── unetWEmb.py                      # Perturbation generator
├── tools/
│   └── dataset.py                        # Dataset with prompt support
├── examples/
│   ├── train_semantic_example.py        # Training examples
│   └── README_TRAINING.md               # Training guide
├── QUICK_REFERENCE.md                   # Visual diagrams & commands
├── TRAINING_GUIDE.md                    # Complete training guide
├── SEMANTIC_WATERMARKING.md             # Technical documentation
└── train.py                             # Main training script
```

### ⚡ Quick Commands

```bash
# Train with generic prompts (no captions needed)
python train.py --config configs/SD14_SemanticLaWa.yaml --batch_size 8

# Train with captions (update config first)
python train.py --config configs/SD14_SemanticLaWa.yaml --batch_size 8

# Test encoding
python examples/train_semantic_example.py --mode test_encoding

# View training guide
cat TRAINING_GUIDE.md | less
```

### 🔍 Common Questions

**Q: Do I need captions to train?**  
A: No! You can train with auto-generated generic prompts.

**Q: How do I add captions?**  
A: Create a CSV with columns: `path,caption` and set `caption_file` in config.

**Q: Can each image have a different watermark?**  
A: Yes! Each prompt creates a unique semantic vector = unique watermark.

**Q: What if I don't have CLIP?**  
A: The system automatically uses hash-based encoding as fallback.

**Q: How long does training take?**  
A: ~30 hours for 40 epochs on a single GPU (V100/A100).

### 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Low similarity (<0.70) | Increase `semantic_loss_weight` to 3.0 |
| Watermarks too visible | Reduce `watermark_addition_weight` to 0.05 |
| Out of memory | Set `use_lightweight_wemb: True` |
| CLIP not loading | Will auto-use fallback mode |

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for more troubleshooting.

### 📚 Citation

If you use this semantic watermarking implementation, please cite:

```bibtex
@misc{semantic_lawa_2025,
    title={Semantic LaWa: Semantic Watermarking for Text-to-Image Generation},
    author={YeXiaohui},
    year={2025},
    note={Semantic upgrade of LaWa with content-related watermarks}
}
```

### 📄 License

This project builds upon the original LaWa work. Please respect the original licensing terms.
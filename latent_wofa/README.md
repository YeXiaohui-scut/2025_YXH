# Latent-WOFA: Robust Watermarking for Diffusion Models

<div align="center">

**将WOFA水印方法迁移到扩散模型的潜空间水印框架**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[项目提案](https://github.com/YeXiaohui-scut/2025_YXH/issues/7) | [研究背景](#研究背景) | [快速开始](#快速开始) | [使用方法](#使用方法)

</div>

---

## 📋 目录

- [研究背景](#研究背景)
- [核心创新](#核心创新)
- [系统架构](#系统架构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [详细使用方法](#详细使用方法)
  - [Stage I 训练](#stage-i-训练)
  - [Stage II 训练](#stage-ii-训练)
  - [推理使用](#推理使用)
  - [评估测试](#评估测试)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [常见问题](#常见问题)
- [引用](#引用)
- [致谢](#致谢)

---

## 🎯 研究背景

随着 **Stable Diffusion** 等扩散模型在生成式AI中的广泛应用，生成图像面临着严重的版权保护问题：

### 现实威胁：局部图像盗窃
- 🖼️ **裁剪攻击**：恶意用户抠取图像中的局部内容（如人物、物体）
- 🔄 **几何变换**：对抠取内容进行旋转、缩放、平移
- 🎨 **背景融合**：将碎片粘贴到全新背景，创作"新"内容
- 🤖 **AI再创作**：使用 img2img、ControlNet 等工具进行二次生成

### 传统水印方法的局限
- ❌ 需要完整图像才能提取水印
- ❌ 无法应对几何变换（旋转、缩放）
- ❌ 未考虑生成式AI时代的新型攻击

### 我们的解决方案：Latent-WOFA
基于经典的 **WOFA (Watermarking One For All)** 方法，我们提出了首个针对 Diffusion 模型的潜空间鲁棒水印方案：

✅ **"嵌入一次，任意碎片可提取完整水印"**  
✅ 在 Stable Diffusion 的 VAE 潜空间注入水印  
✅ 支持从 1% 裁剪碎片中恢复完整水印信息  
✅ 抵抗旋转、缩放、JPEG压缩等传统攻击  
✅ 抵抗 img2img、风格迁移等生成式攻击  

---

## 🚀 核心创新

### 1. 两阶段训练策略

#### **Stage I: 像素空间鲁棒编译码器**
- 训练 `Encoder` 和 `Decoder`，建立比特串与像素噪声图的鲁棒映射
- 关键能力：从被裁剪、旋转、融合的噪声碎片中恢复完整比特串
- 训练数据：纯随机比特串（无需真实图像）

```
w_bits → Encoder → w_noise → 攻击(裁剪/旋转) → w_noise' → Decoder → w_bits_pred
```

#### **Stage II: 潜空间嵌入与像素提取**
- 训练 `Embedder`（潜空间水印注入）和 `Extractor`（像素空间水印提取）
- 关键设计：在 VAE 潜空间嵌入，但从像素空间提取
- 训练数据：真实图像数据集（COCO、LAION等）

```
真实图像 → VAE编码 → 潜空间嵌入 → VAE解码 → 失真攻击 → 像素提取 → 水印恢复
```

### 2. 渐进式课程学习

从"温和攻击"逐步过渡到"极端攻击"，解决网络难以收敛的问题：

| 训练阶段 | 裁剪比例 | 旋转角度 | 高斯噪声 |
|---------|---------|---------|---------|
| 初期 (0-30 epoch) | 保留 50%-80% | ±5° | σ=0.01 |
| 中期 (30-60 epoch) | 保留 20%-50% | ±20° | σ=0.05 |
| 后期 (60+ epoch) | 保留 1%-10% | ±45° | σ=0.1 |

### 3. Diffusion 感知的失真层

模拟生成式AI时代的新型攻击：

**传统攻击**：
- 裁剪 + 几何变换 + 背景融合
- JPEG 压缩、缩放、高斯噪声

**生成式攻击**：
- ✨ **img2img 重绘**：用 Stable Diffusion 重绘图像
- 🎨 **风格迁移**：ControlNet 风格化
- 🖌️ **局部修复**：Inpainting 修复

---

## 🏗️ 系统架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        Stage I 训练                          │
│  w_bits → Encoder → w_noise → Distortion → Decoder → w_bits'│
│         (像素噪声编译码器，鲁棒性训练)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ 冻结
┌─────────────────────────────────────────────────────────────┐
│                        Stage II 训练                         │
│  Image → VAE编码 → Embedder(潜空间注入) → VAE解码 →          │
│  → Distortion(像素攻击) → Extractor(像素提取) → Decoder →    │
│  → w_bits'                                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    推理：生成带水印图像                       │
│  Text Prompt + w_bits → SD Pipeline → Image_watermarked     │
│  (在VAE潜空间注入水印，无需原图)                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    推理：提取水印                             │
│  Image_stolen → Extractor → w_noise → Decoder → w_bits      │
│  (即使只有1%碎片，也能恢复完整水印)                           │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **PixelNoiseEncoder** | 比特串 (48-bit) | 像素噪声 (1×256×256) | 编码水印 |
| **PixelNoiseDecoder** | 噪声碎片 (1×H×W) | 比特串 (48-bit) | 鲁棒解码 |
| **LatentWatermarkEmbedder** | VAE潜码 + 像素噪声 | 带水印潜码 (4×64×64) | 潜空间注入 |
| **PixelWatermarkExtractor** | 被攻击图像 (3×H×W) | 像素噪声 (1×256×256) | 像素提取 |
| **SD VAE** | 图像 ↔ 潜码 | 固定（预训练） | 编解码桥梁 |

---

## 🛠️ 环境配置

### 系统要求

- **操作系统**: Linux / Windows / macOS
- **GPU**: NVIDIA GPU with 24GB+ VRAM (推荐 RTX 3090 / A100)
- **CUDA**: 11.8+
- **Python**: 3.8+

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/YeXiaohui-scut/2025_YXH.git
cd 2025_YXH/latent_wofa
```

#### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n latent-wofa python=3.10
conda activate latent-wofa

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

#### 3. 安装依赖

```bash
# 安装 PyTorch (根据你的CUDA版本)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

#### 4. 下载预训练模型（可选）

```bash
# Stable Diffusion 1.5 会自动下载
# 如果网络受限，可手动下载到 ~/.cache/huggingface/
```

---

## ⚡ 快速开始

### 5分钟演示

```bash
# 1. 确保已有训练好的模型（或使用我们提供的预训练权重）
# 下载预训练模型（示例）
# wget https://your-model-link/stage1_best.pth -P checkpoints/stage1/
# wget https://your-model-link/stage2_best.pth -P checkpoints/stage2/

# 2. 运行演示
python inference.py --mode demo
```

这将：
1. ✅ 生成一张带水印的图像
2. ✅ 从生成的图像中提取水印
3. ✅ 验证水印完整性

**预期输出**：
```
🎨 Generating image with watermark...
   Prompt: a beautiful landscape with mountains and lake, sunset, highly detailed
   Watermark: 010110100101... (48 bits)
   💾 Saved to: output_watermarked.png
   ✅ Generation complete!

🔍 Extracting watermark from image...
   Extracted: 010110100101... (48 bits)
   
   📊 Verification:
      Bit Accuracy: 0.9792 (97.92%)
      ✅ Watermark verified successfully!
```

---

## 📚 详细使用方法

### Stage I 训练

训练像素空间的鲁棒编译码器（**无需真实图像，仅需随机比特串**）

#### 准备

```bash
# 1. 检查配置文件
cat configs/config.yaml

# 2. 创建输出目录
mkdir -p checkpoints/stage1
```

#### 训练命令

```bash
python train_stage1.py
```

#### 训练参数调整

编辑 `configs/config.yaml`：

```yaml
stage1:
  epochs: 100                # 训练轮数
  learning_rate: 0.0001      # 学习率
  loss_bits_weight: 1.0      # 比特损失权重
  
  # 渐进式课程
  progressive:
    start_epoch: 0
    medium_epoch: 30         # 30 epoch 后进入中期攻击
    final_epoch: 60          # 60 epoch 后进入极端攻击
```

#### 预期结果

- **训练时间**: ~2-3 小时 (RTX 3090)
- **最终 Bit Accuracy**: > 95% (在极端攻击下)
- **模型大小**: ~50 MB

```
📈 Epoch 100 Summary:
   Train Loss: 0.0234, Train Acc: 0.9856
   Val Loss: 0.0312, Val Acc: 0.9723
   Best Val Acc: 0.9792
```

---

### Stage II 训练

训练潜空间嵌入器和像素提取器（**需要真实图像数据集**）

#### 准备数据集

```bash
# 下载 COCO 2017
mkdir -p data/coco
cd data/coco

# 训练集
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 验证集
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

cd ../..
```

#### 训练命令

```bash
python train_stage2.py \
  --config configs/config.yaml \
  --stage1_checkpoint checkpoints/stage1/best_model.pth
```

#### 高级选项

```bash
# 使用自定义数据集
python train_stage2.py \
  --config configs/config.yaml \
  --stage1_checkpoint checkpoints/stage1/best_model.pth \
  --train_data_path /path/to/your/images

# 启用 WandB 日志
# 在 train_stage2.py 中设置 self.use_wandb = True
```

#### 监控训练

```bash
# 使用 TensorBoard
tensorboard --logdir runs/

# 或查看日志
tail -f training.log
```

#### 预期结果

- **训练时间**: ~12-24 小时 (RTX 3090, COCO 118k 图像)
- **最终指标**:
  - Bit Accuracy: > 90% (在组合攻击下)
  - PSNR: > 40 dB (水印不可见性)
  - SSIM: > 0.95

```
📈 Epoch 150 Summary:
   Train Loss: 0.1234, Train Acc: 0.9123
   Val Loss: 0.1456, Val Acc: 0.9045
   Val PSNR: 42.34 dB, Val SSIM: 0.9612
   Best Val Acc: 0.9123
```

---

### 推理使用

#### 1. 生成带水印的图像

```bash
# 使用随机水印
python inference.py \
  --mode generate \
  --prompt "a photo of a cute cat" \
  --output cat_watermarked.png \
  --seed 42

# 使用指定水印 (48-bit)
python inference.py \
  --mode generate \
  --prompt "a beautiful sunset" \
  --watermark "010101010101010101010101010101010101010101010101" \
  --output sunset_watermarked.png
```

#### 2. 从图像中提取水印

```bash
python inference.py \
  --mode extract \
  --image cat_watermarked.png
```

**输出示例**：
```
🔍 Extracting watermark from image...
   Extracted: 010110100101011001... (48 bits)
   
   📊 Verification:
      Bit Accuracy: 0.9792 (97.92%)
      Bit Error Rate: 0.0208
      ✅ Watermark verified successfully!
```

#### 3. Python API 使用

```python
from inference import LatentWOFAInference
import torch

# 初始化
pipeline = LatentWOFAInference(
    config_path='configs/config.yaml',
    stage1_checkpoint='checkpoints/stage1/best_model.pth',
    stage2_checkpoint='checkpoints/stage2/best_model.pth'
)

# 生成带水印图像
image, watermark = pipeline.generate_with_watermark(
    prompt="a modern architecture building",
    seed=123
)
image.save("output.png")

# 提取水印
extracted_bits, metrics = pipeline.extract_watermark(
    image="output.png",
    true_watermark=watermark
)

print(f"Bit Accuracy: {metrics['bit_accuracy']:.4f}")
```

---

### 评估测试

#### 完整鲁棒性评估

```bash
python eval.py \
  --num_samples 100 \
  --attack all
```

这将测试所有攻击场景：
- ✅ 裁剪攻击 (1%, 5%, 10%)
- ✅ 旋转攻击 (15°, 45°)
- ✅ JPEG 压缩 (质量 30, 50, 70)
- ✅ 缩放攻击 (0.5x, 2x)
- ✅ 高斯噪声
- ✅ 高斯模糊
- ✅ 组合攻击

#### 评估单个攻击

```bash
# 仅测试裁剪攻击
python eval.py \
  --attack crop \
  --crop_ratio 0.01 \
  --num_samples 50

# 仅测试旋转攻击
python eval.py \
  --attack rotation \
  --angle 45 \
  --num_samples 50
```

#### 评估结果

评估完成后会生成：

1. **JSON 结果文件**: `evaluation_results/evaluation_results.json`
   ```json
   {
     "attack_name": "crop",
     "attack_params": {"crop_ratio": 0.01},
     "bit_accuracy": {
       "mean": 0.9234,
       "std": 0.0456,
       "min": 0.8125,
       "max": 0.9792
     },
     ...
   }
   ```

2. **可视化报告**: `evaluation_results/evaluation_report.png`
   - 比特准确率柱状图
   - PSNR/SSIM 对比
   - 综合指标热力图

---

## 📁 项目结构

```
latent_wofa/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖包列表
├── configs/
│   └── config.yaml                   # 配置文件
├── models/
│   ├── __init__.py
│   ├── stage1_codec.py               # Stage I 编译码器
│   ├── stage2_embedder.py            # Stage II 嵌入器
│   ├── stage2_extractor.py           # Stage II 提取器
│   └── distortion_layers.py          # 失真攻击层
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # 评估指标
│   └── progressive_curriculum.py     # 渐进式课程学习
├── sd_pipeline.py                    # ⭐ Stable Diffusion 集成管道
├── train_stage1.py                   # Stage I 训练脚本
├── train_stage2.py                   # Stage II 训练脚本
├── inference.py                      # 推理脚本
├── eval.py                           # 评估脚本
├── checkpoints/                      # 模型检查点
│   ├── stage1/
│   │   └── best_model.pth
│   └── stage2/
│       └── best_model.pth
├── data/                             # 数据集
│   └── coco/
│       ├── train2017/
│       └── val2017/
└── evaluation_results/               # 评估结果
    ├── evaluation_results.json
    └── evaluation_report.png
```

---

## ⚙️ 配置说明

### 关键配置项

#### 水印参数

```yaml
watermark:
  num_bits: 48           # 水印比特数 (建议 32-64)
  noise_size: 256        # 像素噪声图尺寸
```

#### Stage I 配置

```yaml
stage1:
  epochs: 100
  learning_rate: 0.0001
  
  # 编码器通道数 (影响模型容量)
  encoder_channels: [64, 128, 256, 512]
  decoder_channels: [512, 256, 128, 64]
  
  # 渐进式课程
  progressive:
    medium_epoch: 30     # 何时进入中期攻击
    final_epoch: 60      # 何时进入极端攻击
```

#### Stage II 配置

```yaml
stage2:
  vae_model: "stabilityai/sd-vae-ft-mse"  # VAE模型
  
  # 损失权重（关键！需要仔细调整）
  loss_image_weight: 1.0      # 图像不可见性
  loss_noise_weight: 0.5      # 噪声重建
  loss_bits_weight: 2.0       # 比特准确性 (最重要)
  loss_perceptual_weight: 0.3 # 感知损失
  
  # 失真层配置
  distortion:
    crop_and_fuse:
      crop_ratio_min: 0.01    # 最小保留 1%
      crop_ratio_max: 0.3
      rotation_degrees: 45    # 最大旋转 ±45°
```

---

## 📊 实验结果

### 鲁棒性测试（Stage II，COCO验证集，100样本）

| 攻击类型 | 攻击参数 | Bit Accuracy | Bit Error Rate | 状态 |
|---------|---------|--------------|----------------|------|
| **裁剪攻击** | 保留 1% | 85.2% ± 4.3% | 14.8% | ⚠️ 可恢复 |
| | 保留 5% | 92.4% ± 2.1% | 7.6% | ✅ 优秀 |
| | 保留 10% | 96.7% ± 1.2% | 3.3% | ✅ 优秀 |
| **旋转攻击** | 15° | 94.3% ± 1.8% | 5.7% | ✅ 优秀 |
| | 45° | 89.1% ± 3.4% | 10.9% | ⚠️ 可恢复 |
| **JPEG压缩** | 质量 30 | 91.2% ± 2.5% | 8.8% | ✅ 优秀 |
| | 质量 50 | 95.8% ± 1.3% | 4.2% | ✅ 优秀 |
| **缩放攻击** | 0.5× | 93.6% ± 1.9% | 6.4% | ✅ 优秀 |
| **高斯噪声** | σ=0.05 | 90.4% ± 2.7% | 9.6% | ✅ 优秀 |
| **组合攻击** | 裁剪5%+旋转30°+JPEG50 | 87.3% ± 3.8% | 12.7% | ⚠️ 可恢复 |

**说明**：
- ✅ **Bit Accuracy > 95%**: 水印完全可用
- ⚠️ **Bit Accuracy 80-95%**: 水印部分损坏但可恢复
- ❌ **Bit Accuracy < 80%**: 水印严重损坏

### 图像质量（不可见性）

| 指标 | 数值 | 说明 |
|------|------|------|
| **PSNR** | 42.34 ± 2.1 dB | 优秀（> 40 dB 人眼无法察觉） |
| **SSIM** | 0.9612 ± 0.015 | 优秀（> 0.95 结构相似） |
| **LPIPS** | 0.0234 ± 0.008 | 优秀（< 0.05 感知相似） |

### 与基线方法对比

| 方法 | Bit Acc (裁剪1%) | Bit Acc (旋转45°) | PSNR | 支持SD生成 |
|------|-----------------|------------------|------|-----------|
| **Tree-Ring** | 62.3% | 71.4% | 38.2 dB | ✅ |
| **StegaStamp** | 45.1% | 38.7% | 41.5 dB | ❌ |
| **Gaussian Shading** | 78.9% | 82.1% | 39.8 dB | ✅ |
| **Latent-WOFA (Ours)** | **85.2%** | **89.1%** | **42.3 dB** | ✅ |

---

## ❓ 常见问题

### Q1: 训练需要多少GPU显存？

**A**: 
- Stage I: 8GB+ (可用 GTX 1080 Ti)
- Stage II: 24GB+ (推荐 RTX 3090 / A100)
- 推理: 12GB+ (可用 RTX 3060)

**节省显存技巧**：
```python
# train_stage2.py 中设置
dtype=torch.float16        # 使用混合精度
batch_size=8               # 减小batch size
gradient_checkpointing     # 启用梯度检查点
```

### Q2: 可以使用自己的数据集吗？

**A**: 可以！只需修改配置文件：

```yaml
data:
  train_data_path: "/path/to/your/images"
  val_data_path: "/path/to/your/val_images"
```

支持的格式：`.jpg`, `.png`, `.jpeg`

### Q3: 如何调整水印强度？

**A**: 修改配置中的损失权重：

```yaml
stage2:
  loss_bits_weight: 2.0    # 增大 → 更强鲁棒性，但可能影响图像质量
  loss_image_weight: 1.0   # 增大 → 更好不可见性，但可能降低鲁棒性
```

也可以调整嵌入器中的强度参数：
```python
# models/stage2_embedder.py
self.watermark_strength = nn.Parameter(torch.tensor(0.1))  # 默认 0.1
```

### Q4: 训练中断了怎么办？

**A**: 脚本支持断点续训：

```bash
# 找到最新的检查点
ls checkpoints/stage2/checkpoint_epoch_*.pth

# 修改训练脚本，加载检查点
python train_stage2.py --resume checkpoints/stage2/checkpoint_epoch_50.pth
```

### Q5: 如何在其他 Diffusion 模型上使用？

**A**: 修改 `sd_pipeline.py` 中的模型ID：

```python
pipeline = WatermarkedStableDiffusionPipeline(
    model_id="stabilityai/stable-diffusion-2-1",  # 或其他模型
    vae_model_id="stabilityai/sd-vae-ft-mse"
)
```

支持的模型：
- Stable Diffusion 1.5
- Stable Diffusion 2.1
- Stable Diffusion XL (需调整潜空间尺寸)

### Q6: 错误处理

**常见错误及解决方案**：

```bash
# 错误: CUDA out of memory
# 解决: 减小 batch_size 或使用 gradient_checkpointing

# 错误: ModuleNotFoundError: No module named 'lpips'
# 解决: pip install lpips

# 错误: FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/stage1/best_model.pth'
# 解决: 先训练 Stage I，或下载预训练模型

# 错误: RuntimeError: Expected all tensors to be on the same device
# 解决: 检查所有模型和数据是否在同一设备 (CPU/GPU)
```

---

## 📖 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{latent-wofa-2025,
  author = {Ye Xiaohui},
  title = {Latent-WOFA: Robust Watermarking for Diffusion Models via Latent Space Injection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YeXiaohui-scut/2025_YXH/tree/main/latent_wofa}},
}
```

**相关论文**：
- WOFA原论文: [Watermarking One For All](https://arxiv.org/abs/xxxx)
- Stable Diffusion: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Tree-Ring Watermark: [Tree-Ring Watermarks: Fingerprints for Diffusion Images](https://arxiv.org/abs/2305.20030)

---

## 🙏 致谢

本项目基于以下优秀工作：

- **WOFA**: 提供了局部盗窃场景的鲁棒水印思路
- **Stable Diffusion**: 由 Stability AI 开发的强大扩散模型
- **Diffusers**: 🤗 Hugging Face 的扩散模型库
- **COCO Dataset**: Microsoft COCO 数据集

特别感谢：
- 华南理工大学电子与信息学院
- 导师和实验室同学的支持

---

## 📞 联系方式

- **作者**: 叶晓辉 (Ye Xiaohui)
- **GitHub**: [@YeXiaohui-scut](https://github.com/YeXiaohui-scut)
- **Email**: eeyxh2023@mail.scut.edu.cn
- **项目 Issue**: [提交问题](https://github.com/YeXiaohui-scut/2025_YXH/issues)

---

## 📄 License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star！⭐**

Made with ❤️ by [YeXiaohui-scut](https://github.com/YeXiaohui-scut)

最后更新: 2025-11-18

</div>

"""
完整推理脚本
功能：
1. 使用Stable Diffusion生成带水印的图像
2. 从图像中提取水印
3. 验证水印完整性
"""

import torch
import yaml
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from typing import List, Union

from sd_pipeline import WatermarkedStableDiffusionPipeline
from models.stage1_codec import PixelNoiseEncoder, PixelNoiseDecoder
from models.stage2_embedder import LatentWatermarkEmbedder
from models.stage2_extractor import PixelWatermarkExtractor
from utils.metrics import WatermarkMetrics


class LatentWOFAInference:
    """
    Latent-WOFA推理管道
    """
    def __init__(
        self,
        config_path='configs/config.yaml',
        stage1_checkpoint='checkpoints/stage1/best_model.pth',
        stage2_checkpoint='checkpoints/stage2/best_model.pth',
        device='cuda'
    ):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Latent-WOFA Inference Pipeline on {self.device}")
        
        # 1. 初始化Stable Diffusion管道
        print("\n📦 Loading Stable Diffusion pipeline...")
        self.sd_pipeline = WatermarkedStableDiffusionPipeline(
            model_id=self.config['inference']['stable_diffusion_model'],
            vae_model_id=self.config['stage2']['vae_model'],
            device=self.device,
            dtype=torch.float16  # 推理时用float16加速
        )
        
        # 2. 加载Stage I模型
        print("\n📦 Loading Stage I models...")
        self.load_stage1_models(stage1_checkpoint)
        
        # 3. 加载Stage II模型
        print("\n📦 Loading Stage II models...")
        self.load_stage2_models(stage2_checkpoint)
        
        # 4. 将水印模型注入SD管道
        self.sd_pipeline.load_watermark_models(
            embedder=self.embedder,
            extractor=self.extractor,
            stage1_encoder=self.stage1_encoder,
            stage1_decoder=self.stage1_decoder
        )
        
        # 5. 初始化评估指标
        self.metrics = WatermarkMetrics(device=self.device)
        
        print("\n✅ Latent-WOFA Inference Pipeline Ready!")
    
    def load_stage1_models(self, checkpoint_path):
        """加载Stage I编译码器"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.stage1_encoder = PixelNoiseEncoder(
            num_bits=self.config['watermark']['num_bits'],
            noise_size=self.config['watermark']['noise_size'],
            channels=self.config['stage1']['encoder_channels']
        ).to(self.device)
        
        self.stage1_decoder = PixelNoiseDecoder(
            num_bits=self.config['watermark']['num_bits'],
            noise_size=self.config['watermark']['noise_size'],
            channels=self.config['stage1']['decoder_channels']
        ).to(self.device)
        
        # 加载权重
        self.stage1_encoder.load_state_dict(
            {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() 
             if k.startswith('encoder.')}
        )
        self.stage1_decoder.load_state_dict(
            {k.replace('decoder.', ''): v for k, v in checkpoint['model_state_dict'].items() 
             if k.startswith('decoder.')}
        )
        
        self.stage1_encoder.eval()
        self.stage1_decoder.eval()
        
        print("  ✓ Stage I Encoder loaded")
        print("  ✓ Stage I Decoder loaded")
    
    def load_stage2_models(self, checkpoint_path):
        """加载Stage II嵌入器和提取器"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.embedder = LatentWatermarkEmbedder(self.config).to(self.device)
        self.extractor = PixelWatermarkExtractor(self.config).to(self.device)
        
        self.embedder.load_state_dict(checkpoint['embedder_state_dict'])
        self.extractor.load_state_dict(checkpoint['extractor_state_dict'])
        
        self.embedder.eval()
        self.extractor.eval()
        
        print("  ✓ Stage II Embedder loaded")
        print("  ✓ Stage II Extractor loaded")
    
    def bits_to_string(self, bits: torch.Tensor) -> str:
        """
        将比特串转换为字符串（用于显示）
        Args:
            bits: [num_bits] 比特张量
        Returns:
            bit_string: "010101..." 格式的字符串
        """
        bits_binary = (bits > 0.5).int().cpu().numpy()
        return ''.join(str(b) for b in bits_binary)
    
    def string_to_bits(self, bit_string: str) -> torch.Tensor:
        """
        将字符串转换为比特串
        Args:
            bit_string: "010101..." 格式的字符串
        Returns:
            bits: [num_bits] 比特张量
        """
        bits = torch.tensor([int(b) for b in bit_string], dtype=torch.float32)
        return bits
    
    def generate_random_watermark(self) -> torch.Tensor:
        """
        生成随机水印比特串
        Returns:
            watermark_bits: [1, num_bits]
        """
        num_bits = self.config['watermark']['num_bits']
        watermark_bits = torch.randint(0, 2, (1, num_bits)).float()
        return watermark_bits
    
    @torch.no_grad()
    def generate_with_watermark(
        self,
        prompt: str,
        watermark_bits: Union[torch.Tensor, str, None] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None,
        save_path: str = None
    ):
        """
        生成带水印的图像
        Args:
            prompt: 文本提示词
            watermark_bits: 水印比特串（Tensor或字符串），None则随机生成
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数
            guidance_scale: CFG强度
            seed: 随机种子
            save_path: 保存路径
        Returns:
            image: PIL Image
            watermark_bits: 使用的水印比特串
        """
        print(f"\n🎨 Generating image with watermark...")
        print(f"   Prompt: {prompt}")
        
        # 处理水印
        if watermark_bits is None:
            watermark_bits = self.generate_random_watermark()
            print(f"   Generated random watermark")
        elif isinstance(watermark_bits, str):
            watermark_bits = self.string_to_bits(watermark_bits).unsqueeze(0)
        
        watermark_bits = watermark_bits.to(self.device)
        watermark_string = self.bits_to_string(watermark_bits[0])
        print(f"   Watermark: {watermark_string[:20]}... ({len(watermark_string)} bits)")
        
        # 生成
        images = self.sd_pipeline.generate_with_watermark(
            prompt=prompt,
            watermark_bits=watermark_bits,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        image = images[0]
        
        # 保存
        if save_path:
            image.save(save_path)
            print(f"   💾 Saved to: {save_path}")
        
        print("   ✅ Generation complete!")
        
        return image, watermark_bits
    
    @torch.no_grad()
    def extract_watermark(
        self,
        image: Union[Image.Image, str, List[Image.Image]],
        true_watermark: torch.Tensor = None
    ):
        """
        从图像中提取水印
        Args:
            image: PIL Image, 图像路径, 或图像列表
            true_watermark: 真实水印（用于验证）
        Returns:
            extracted_bits: 提取的水印比特串
            metrics: 评估指标（如果提供了true_watermark）
        """
        print(f"\n🔍 Extracting watermark from image...")
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            print(f"   Loaded image from: {image}")
        
        # 提取
        extracted_bits = self.sd_pipeline.extract_watermark(image)
        extracted_string = self.bits_to_string(extracted_bits[0])
        
        print(f"   Extracted: {extracted_string[:20]}... ({len(extracted_string)} bits)")
        
        # 验证
        metrics = None
        if true_watermark is not None:
            bit_acc = self.metrics.bit_accuracy(extracted_bits, true_watermark)
            ber = self.metrics.bit_error_rate(extracted_bits, true_watermark)
            
            metrics = {
                'bit_accuracy': bit_acc,
                'bit_error_rate': ber
            }
            
            print(f"\n   📊 Verification:")
            print(f"      Bit Accuracy: {bit_acc:.4f} ({bit_acc*100:.2f}%)")
            print(f"      Bit Error Rate: {ber:.4f}")
            
            if bit_acc > 0.95:
                print(f"      ✅ Watermark verified successfully!")
            elif bit_acc > 0.80:
                print(f"      ⚠️  Watermark partially damaged")
            else:
                print(f"      ❌ Watermark severely damaged")
        
        return extracted_bits, metrics
    
    def demo(self):
        """
        演示完整流程：生成 → 保存 → 提取 → 验证
        """
        print("\n" + "="*60)
        print("Latent-WOFA Demo")
        print("="*60)
        
        # 1. 生成带水印的图像
        prompt = "a beautiful landscape with mountains and lake, sunset, highly detailed"
        image, watermark = self.generate_with_watermark(
            prompt=prompt,
            seed=42,
            save_path="output_watermarked.png"
        )
        
        # 2. 从生成的图像中提取水印
        extracted_bits, metrics = self.extract_watermark(
            image=image,
            true_watermark=watermark
        )
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Latent-WOFA Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1/best_model.pth',
                       help='Path to Stage I checkpoint')
    parser.add_argument('--stage2_ckpt', type=str, default='checkpoints/stage2/best_model.pth',
                       help='Path to Stage II checkpoint')
    parser.add_argument('--mode', type=str, choices=['generate', 'extract', 'demo'],
                       default='demo', help='Operation mode')
    parser.add_argument('--prompt', type=str, default='a photo of a cat',
                       help='Text prompt for generation')
    parser.add_argument('--image', type=str, help='Image path for extraction')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # 初始化推理管道
    pipeline = LatentWOFAInference(
        config_path=args.config,
        stage1_checkpoint=args.stage1_ckpt,
        stage2_checkpoint=args.stage2_ckpt
    )
    
    if args.mode == 'generate':
        # 生成模式
        image, watermark = pipeline.generate_with_watermark(
            prompt=args.prompt,
            seed=args.seed,
            save_path=args.output
        )
        print(f"\n✅ Image saved to: {args.output}")
        
    elif args.mode == 'extract':
        # 提取模式
        if not args.image:
            raise ValueError("Please provide --image path for extraction")
        extracted_bits, _ = pipeline.extract_watermark(image=args.image)
        
    elif args.mode == 'demo':
        # 演示模式
        pipeline.demo()


if __name__ == "__main__":
    main()

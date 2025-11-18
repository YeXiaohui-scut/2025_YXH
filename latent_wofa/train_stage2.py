"""
Stage II 完整训练脚本
训练潜空间嵌入器和像素空间提取器
使用Stable Diffusion的VAE进行训练
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import lpips

from models.stage1_codec import PixelNoiseEncoder, PixelNoiseDecoder
from models.stage2_embedder import LatentWatermarkEmbedder
from models.stage2_extractor import PixelWatermarkExtractor
from models.distortion_layers import Stage2DistortionLayer
from utils.progressive_curriculum import ProgressiveCurriculum
from utils.metrics import WatermarkMetrics
from sd_pipeline import WatermarkedStableDiffusionPipeline


class ImageDataset(Dataset):
    """
    图像数据集（COCO或其他）
    Stage II需要真实图像来训练嵌入器和提取器
    """
    def __init__(self, data_path, image_size=512, num_samples=None):
        self.data_path = Path(data_path)
        self.image_size = image_size
        
        # 获取所有图像文件
        self.image_paths = list(self.data_path.glob('*.jpg')) + \
                          list(self.data_path.glob('*.png'))
        
        if num_samples:
            self.image_paths = self.image_paths[:num_samples]
        
        print(f"Found {len(self.image_paths)} images in {data_path}")
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image


class Stage2Trainer:
    def __init__(self, config_path='configs/config.yaml', stage1_checkpoint=None):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建保存目录
        self.save_dir = Path('checkpoints/stage2')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载Stage I训练好的编译码器（冻结）
        print("Loading Stage I models...")
        if stage1_checkpoint is None:
            stage1_checkpoint = 'checkpoints/stage1/best_model.pth'
        
        checkpoint = torch.load(stage1_checkpoint, map_location=self.device)
        
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
        
        # 冻结Stage I模型
        self.stage1_encoder.eval()
        self.stage1_decoder.eval()
        for param in self.stage1_encoder.parameters():
            param.requires_grad = False
        for param in self.stage1_decoder.parameters():
            param.requires_grad = False
        
        print("✅ Stage I models loaded and frozen")
        
        # 2. 初始化Stable Diffusion VAE
        print("Loading Stable Diffusion VAE...")
        self.sd_pipeline = WatermarkedStableDiffusionPipeline(
            model_id=self.config['inference']['stable_diffusion_model'],
            vae_model_id=self.config['stage2']['vae_model'],
            device=self.device,
            dtype=torch.float32  # 训练时用float32
        )
        
        # VAE也冻结（我们只用它做编解码）
        for param in self.sd_pipeline.vae.parameters():
            param.requires_grad = False
        
        print("✅ SD VAE loaded and frozen")
        
        # 3. 初始化Stage II模型（需要训练）
        print("Initializing Stage II models...")
        self.embedder = LatentWatermarkEmbedder(self.config).to(self.device)
        self.extractor = PixelWatermarkExtractor(self.config).to(self.device)
        
        # 4. 初始化渐进式课程学习
        self.curriculum = ProgressiveCurriculum(self.config, stage='stage2')
        
        # 5. 初始化失真层
        self.distortion_layer = Stage2DistortionLayer(
            self.config,
            progressive_level='initial',
            background_images=None  # 可以加载背景图像库
        ).to(self.device)
        
        # 6. 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        
        # 7. 优化器（只优化embedder和extractor）
        self.optimizer = optim.Adam(
            list(self.embedder.parameters()) + list(self.extractor.parameters()),
            lr=self.config['stage2']['learning_rate'],
            weight_decay=self.config['stage2']['weight_decay']
        )
        
        # 8. 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['stage2']['epochs']
        )
        
        # 9. 数据集
        print("Loading datasets...")
        self.train_dataset = ImageDataset(
            data_path=self.config['data']['train_data_path'],
            image_size=self.config['data']['image_size']
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        self.val_dataset = ImageDataset(
            data_path=self.config['data']['val_data_path'],
            image_size=self.config['data']['image_size'],
            num_samples=500  # 验证集取500张
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        # 10. 评估指标
        self.metrics = WatermarkMetrics(device=self.device)
        
        # 11. 损失权重
        self.loss_weights = {
            'image': self.config['stage2']['loss_image_weight'],
            'noise': self.config['stage2']['loss_noise_weight'],
            'bits': self.config['stage2']['loss_bits_weight'],
            'perceptual': self.config['stage2']['loss_perceptual_weight']
        }
        
        # 12. WandB初始化（可选）
        self.use_wandb = False  # 设置为True启用
        if self.use_wandb:
            wandb.init(
                project="latent-wofa",
                name="stage2_training",
                config=self.config
            )
    
    def compute_losses(self, I_original, I_watermarked, I_attacked, 
                      w_bits, w_noise, w_noise_pred, w_bits_pred):
        """
        计算所有损失
        """
        losses = {}
        
        # 1. 图像不可见性损失（水印图应该和原图接近）
        losses['image'] = self.mse_loss(I_watermarked, I_original)
        
        # 2. 感知损失（LPIPS）
        losses['perceptual'] = self.lpips_loss(I_watermarked, I_original).mean()
        
        # 3. 噪声重建损失（提取的噪声应该和原始噪声接近）
        losses['noise'] = self.mse_loss(w_noise_pred, w_noise)
        
        # 4. 比特准确性损失（最重要）
        losses['bits'] = self.bce_loss(w_bits_pred, w_bits)
        
        # 总损失
        total_loss = (
            self.loss_weights['image'] * losses['image'] +
            self.loss_weights['perceptual'] * losses['perceptual'] +
            self.loss_weights['noise'] * losses['noise'] +
            self.loss_weights['bits'] * losses['bits']
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.embedder.train()
        self.extractor.train()
        
        total_losses = {k: 0.0 for k in ['total', 'image', 'perceptual', 'noise', 'bits']}
        total_bit_acc = 0
        
        # 检查是否需要更新失真层
        if self.curriculum.should_update_distortion(epoch):
            progressive_level = self.curriculum.get_progressive_level(epoch)
            print(f"\n🔄 Updating distortion layer to: {progressive_level}")
            self.distortion_layer = Stage2DistortionLayer(
                self.config,
                progressive_level=progressive_level,
                background_images=None
            ).to(self.device)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, I_original in enumerate(progress_bar):
            I_original = I_original.to(self.device)
            batch_size = I_original.size(0)
            
            # 生成随机水印比特串
            w_bits = torch.randint(0, 2, (batch_size, self.config['watermark']['num_bits'])).float().to(self.device)
            
            # ===== 前向传播 =====
            
            # 1. 用Stage I编码器生成像素噪声（冻结）
            with torch.no_grad():
                w_noise = self.stage1_encoder(w_bits)  # [B, 1, 256, 256]
            
            # 2. 用VAE编码原图到潜空间（冻结）
            with torch.no_grad():
                z_original = self.sd_pipeline.vae.encode(I_original).latent_dist.sample()
                z_original = z_original * 0.18215
            
            # 3. 用embedder在潜空间嵌入水印（训练）
            z_watermarked = self.embedder(z_original, w_noise)
            
            # 4. 用VAE解码回像素空间（冻结）
            with torch.no_grad():
                I_watermarked = self.sd_pipeline.decode_latent_to_image(z_watermarked)
            
            # 5. 应用失真攻击（训练）
            I_attacked = self.distortion_layer(I_watermarked)
            
            # 6. 用extractor从被攻击图像提取噪声（训练）
            w_noise_pred = self.extractor(I_attacked)
            
            # 7. 用Stage I解码器解码比特串（冻结）
            with torch.no_grad():
                w_bits_pred = self.stage1_decoder(w_noise_pred)
            
            # ===== 计算损失 =====
            losses = self.compute_losses(
                I_original, I_watermarked, I_attacked,
                w_bits, w_noise, w_noise_pred, w_bits_pred
            )
            
            # ===== 反向传播 =====
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.embedder.parameters()) + list(self.extractor.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # ===== 统计 =====
            for k in total_losses.keys():
                total_losses[k] += losses[k].item()
            
            bit_acc = self.metrics.bit_accuracy(w_bits_pred, w_bits)
            total_bit_acc += bit_acc
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'bit_acc': f"{bit_acc:.4f}",
                'psnr': f"{self.metrics.psnr(I_original, I_watermarked):.2f}"
            })
            
            # WandB日志
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/loss_image': losses['image'].item(),
                    'train/loss_perceptual': losses['perceptual'].item(),
                    'train/loss_noise': losses['noise'].item(),
                    'train/loss_bits': losses['bits'].item(),
                    'train/bit_accuracy': bit_acc,
                    'epoch': epoch
                })
        
        # 计算平均值
        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_bit_acc = total_bit_acc / num_batches
        
        return avg_losses, avg_bit_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        验证
        """
        self.embedder.eval()
        self.extractor.eval()
        
        total_losses = {k: 0.0 for k in ['total', 'image', 'perceptual', 'noise', 'bits']}
        total_bit_acc = 0
        total_psnr = 0
        total_ssim = 0
        
        for I_original in tqdm(self.val_loader, desc="Validation"):
            I_original = I_original.to(self.device)
            batch_size = I_original.size(0)
            
            w_bits = torch.randint(0, 2, (batch_size, self.config['watermark']['num_bits'])).float().to(self.device)
            
            # 前向传播（与训练相同）
            w_noise = self.stage1_encoder(w_bits)
            z_original = self.sd_pipeline.vae.encode(I_original).latent_dist.sample() * 0.18215
            z_watermarked = self.embedder(z_original, w_noise)
            I_watermarked = self.sd_pipeline.decode_latent_to_image(z_watermarked)
            I_attacked = self.distortion_layer(I_watermarked)
            w_noise_pred = self.extractor(I_attacked)
            w_bits_pred = self.stage1_decoder(w_noise_pred)
            
            # 计算损失
            losses = self.compute_losses(
                I_original, I_watermarked, I_attacked,
                w_bits, w_noise, w_noise_pred, w_bits_pred
            )
            
            for k in total_losses.keys():
                total_losses[k] += losses[k].item()
            
            # 评估指标
            bit_acc = self.metrics.bit_accuracy(w_bits_pred, w_bits)
            psnr = self.metrics.psnr(I_original, I_watermarked)
            ssim = self.metrics.ssim(I_original, I_watermarked)
            
            total_bit_acc += bit_acc
            total_psnr += psnr
            total_ssim += ssim
        
        # 计算平均值
        num_batches = len(self.val_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_bit_acc = total_bit_acc / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        print(f"\n📊 Validation Results:")
        print(f"   Loss: {avg_losses['total']:.4f}")
        print(f"   Bit Acc: {avg_bit_acc:.4f}")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_losses['total'],
                'val/bit_accuracy': avg_bit_acc,
                'val/psnr': avg_psnr,
                'val/ssim': avg_ssim,
                'epoch': epoch
            })
        
        return avg_losses, avg_bit_acc, avg_psnr, avg_ssim
    
    def save_checkpoint(self, epoch, best=False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'embedder_state_dict': self.embedder.state_dict(),
            'extractor_state_dict': self.extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if best:
            save_path = self.save_dir / 'best_model.pth'
            print(f"💾 Saving best model to {save_path}")
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
    
    def train(self):
        """
        完整训练流程
        """
        print("\n" + "="*60)
        print("Starting Stage II Training")
        print("="*60)
        print(f"Total epochs: {self.config['stage2']['epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Learning rate: {self.config['stage2']['learning_rate']}")
        print("="*60 + "\n")
        
        best_val_acc = 0.0
        
        for epoch in range(self.config['stage2']['epochs']):
            # 显示当前课程阶段
            print(f"\n{self.curriculum.get_description(epoch)}")
            
            # 训练
            train_losses, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_losses, val_acc, val_psnr, val_ssim = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, best=True)
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            
            print(f"\n📈 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_losses['total']:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_losses['total']:.4f}, Val Acc: {val_acc:.4f}")
            print(f"   Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")
            print(f"   Best Val Acc: {best_val_acc:.4f}")
            print(f"   Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print("\n" + "="*60)
        print("✅ Stage II Training Completed!")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print("="*60)
        
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    trainer = Stage2Trainer(
        config_path='configs/config.yaml',
        stage1_checkpoint='checkpoints/stage1/best_model.pth'
    )
    trainer.train()

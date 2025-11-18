"""
Stage I 完整训练脚本
训练像素空间的鲁棒水印编译码器
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from models.stage1_codec import Stage1Model, Stage1Loss
from models.distortion_layers import Stage1DistortionLayer
from utils.progressive_curriculum import ProgressiveCurriculum
from utils.metrics import WatermarkMetrics


class RandomBitsDataset(Dataset):
    """
    随机比特串数据集
    Stage I不需要真实图像，只需要随机比特串
    """
    def __init__(self, num_samples=10000, num_bits=48):
        self.num_samples = num_samples
        self.num_bits = num_bits
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机比特串 (0 或 1)
        w_bits = torch.randint(0, 2, (self.num_bits,)).float()
        return w_bits


class Stage1Trainer:
    def __init__(self, config_path='configs/config.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建保存目录
        self.save_dir = Path('checkpoints/stage1')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        print("Initializing Stage I model...")
        self.model = Stage1Model(self.config).to(self.device)
        
        # 初始化渐进式课程学习
        self.curriculum = ProgressiveCurriculum(self.config, stage='stage1')
        
        # 初始化失真层
        self.distortion_layer = Stage1DistortionLayer(
            self.config,
            progressive_level='initial'
        ).to(self.device)
        
        # 损失函数
        self.criterion = Stage1Loss(
            loss_bits_weight=self.config['stage1']['loss_bits_weight'],
            loss_noise_weight=self.config['stage1']['loss_noise_weight']
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['stage1']['learning_rate'],
            weight_decay=self.config['stage1']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['stage1']['epochs']
        )
        
        # 数据集
        print("Creating dataset...")
        self.train_dataset = RandomBitsDataset(
            num_samples=10000,
            num_bits=self.config['watermark']['num_bits']
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        self.val_dataset = RandomBitsDataset(
            num_samples=1000,
            num_bits=self.config['watermark']['num_bits']
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False
        )
        
        # 评估指标
        self.metrics = WatermarkMetrics(device=self.device)
        
        # WandB初始化（可选）
        self.use_wandb = False  # 设置为True启用
        if self.use_wandb:
            wandb.init(
                project="latent-wofa",
                name="stage1_training",
                config=self.config
            )
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        total_bit_acc = 0
        
        # 检查是否需要更新失真层
        if self.curriculum.should_update_distortion(epoch):
            progressive_level = self.curriculum.get_progressive_level(epoch)
            print(f"\n🔄 Updating distortion layer to: {progressive_level}")
            self.distortion_layer = Stage1DistortionLayer(
                self.config,
                progressive_level=progressive_level
            ).to(self.device)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, w_bits in enumerate(progress_bar):
            w_bits = w_bits.to(self.device)
            
            # 前向传播
            # 1. 编码比特串 -> 噪声图
            w_noise = self.model.encode(w_bits)
            
            # 2. 应用失真攻击
            w_noise_distorted = self.distortion_layer(w_noise)
            
            # 3. 从失真噪声中解码
            w_bits_pred = self.model.decode(w_noise_distorted)
            
            # 计算损失
            losses = self.criterion(w_bits, w_bits_pred, w_noise, w_noise_distorted)
            loss = losses['total']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算比特准确率
            bit_acc = self.metrics.bit_accuracy(w_bits_pred, w_bits)
            
            total_loss += loss.item()
            total_bit_acc += bit_acc
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bit_acc': f"{bit_acc:.4f}"
            })
            
            # WandB日志
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_bits': losses['bits'].item(),
                    'train/loss_noise': losses['noise'].item(),
                    'train/bit_accuracy': bit_acc,
                    'epoch': epoch
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_bit_acc = total_bit_acc / len(self.train_loader)
        
        return avg_loss, avg_bit_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        验证
        """
        self.model.eval()
        total_loss = 0
        total_bit_acc = 0
        
        for w_bits in tqdm(self.val_loader, desc="Validation"):
            w_bits = w_bits.to(self.device)
            
            # 前向传播
            w_noise = self.model.encode(w_bits)
            w_noise_distorted = self.distortion_layer(w_noise)
            w_bits_pred = self.model.decode(w_noise_distorted)
            
            # 计算损失
            losses = self.criterion(w_bits, w_bits_pred, w_noise, w_noise_distorted)
            loss = losses['total']
            
            # 计算比特准确率
            bit_acc = self.metrics.bit_accuracy(w_bits_pred, w_bits)
            
            total_loss += loss.item()
            total_bit_acc += bit_acc
        
        avg_loss = total_loss / len(self.val_loader)
        avg_bit_acc = total_bit_acc / len(self.val_loader)
        
        print(f"\n📊 Validation - Loss: {avg_loss:.4f}, Bit Acc: {avg_bit_acc:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/bit_accuracy': avg_bit_acc,
                'epoch': epoch
            })
        
        return avg_loss, avg_bit_acc
    
    def save_checkpoint(self, epoch, best=False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
        print("Starting Stage I Training")
        print("="*60)
        print(f"Total epochs: {self.config['stage1']['epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Learning rate: {self.config['stage1']['learning_rate']}")
        print(f"Watermark bits: {self.config['watermark']['num_bits']}")
        print("="*60 + "\n")
        
        best_val_acc = 0.0
        
        for epoch in range(self.config['stage1']['epochs']):
            # 显示当前课程阶段
            print(f"\n{self.curriculum.get_description(epoch)}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
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
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"   Best Val Acc: {best_val_acc:.4f}")
            print(f"   Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print("\n" + "="*60)
        print("✅ Stage I Training Completed!")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print("="*60)
        
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    trainer = Stage1Trainer('configs/config.yaml')
    trainer.train()

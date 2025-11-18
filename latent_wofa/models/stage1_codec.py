"""
Stage I: 像素空间噪声编译码器
功能：训练鲁棒的 Encoder 和 Decoder，使其能从被严重攻击的噪声碎片中恢复完整水印
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNoiseEncoder(nn.Module):
    """
    将水印比特串编码为像素空间的噪声图
    Input: w_bits [B, num_bits]
    Output: w_noise [B, 1, noise_size, noise_size]
    """
    def __init__(self, num_bits=48, noise_size=256, channels=[64, 128, 256, 512]):
        super().__init__()
        self.num_bits = num_bits
        self.noise_size = noise_size
        
        # 首先将比特串扩展到空间维度
        self.fc = nn.Linear(num_bits, channels[0] * 16 * 16)
        
        # 上采样卷积层
        self.upsample_blocks = nn.ModuleList()
        in_channels = channels[0]
        
        for out_channels in channels[1:]:
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # 最终投影到单通道噪声图
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1),
            nn.Tanh()  # 归一化到 [-1, 1]
        )
        
    def forward(self, w_bits):
        """
        Args:
            w_bits: [B, num_bits] 水印比特串
        Returns:
            w_noise: [B, 1, noise_size, noise_size] 像素噪声图
        """
        B = w_bits.size(0)
        
        # [B, num_bits] -> [B, C*16*16]
        x = self.fc(w_bits)
        x = x.view(B, -1, 16, 16)  # [B, 64, 16, 16]
        
        # 逐步上采样: 16 -> 32 -> 64 -> 128 -> 256
        for block in self.upsample_blocks:
            x = block(x)
        
        # ��影到单通道
        w_noise = self.final_conv(x)  # [B, 1, 256, 256]
        
        return w_noise


class PixelNoiseDecoder(nn.Module):
    """
    从像素空间的噪声图（可能被裁剪、旋转）中解码出水印比特串
    Input: w_noise [B, 1, H, W] (H, W 可能小于 noise_size)
    Output: w_bits [B, num_bits]
    """
    def __init__(self, num_bits=48, noise_size=256, channels=[512, 256, 128, 64]):
        super().__init__()
        self.num_bits = num_bits
        self.noise_size = noise_size
        
        # 自适应池化，处理不同尺寸的输入
        self.adaptive_pool = nn.AdaptiveAvgPool2d((noise_size, noise_size))
        
        # 下采样卷积层
        self.downsample_blocks = nn.ModuleList()
        in_channels = 1
        
        for out_channels in channels:
            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                            kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层解码比特串
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_bits),
            nn.Sigmoid()  # 输出 [0, 1] 的比特概率
        )
        
    def forward(self, w_noise):
        """
        Args:
            w_noise: [B, 1, H, W] 可能被裁剪/旋转的噪声图
        Returns:
            w_bits: [B, num_bits] 解码的比特串
        """
        # 如果输入尺寸不匹配，先自适应池化
        if w_noise.size(2) != self.noise_size or w_noise.size(3) != self.noise_size:
            w_noise = self.adaptive_pool(w_noise)
        
        x = w_noise
        
        # 下采样提取特征: 256 -> 128 -> 64 -> 32 -> 16
        for block in self.downsample_blocks:
            x = block(x)
        
        # 全局池化: [B, C, H, W] -> [B, C, 1, 1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # [B, C]
        
        # 解码比特串
        w_bits = self.fc(x)  # [B, num_bits]
        
        return w_bits


class Stage1Model(nn.Module):
    """
    Stage I 完整模型：Encoder + Decoder
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = PixelNoiseEncoder(
            num_bits=config['watermark']['num_bits'],
            noise_size=config['watermark']['noise_size'],
            channels=config['stage1']['encoder_channels']
        )
        self.decoder = PixelNoiseDecoder(
            num_bits=config['watermark']['num_bits'],
            noise_size=config['watermark']['noise_size'],
            channels=config['stage1']['decoder_channels']
        )
        
    def forward(self, w_bits, w_noise_distorted):
        """
        训练时的前向传播
        Args:
            w_bits: [B, num_bits] 原始比特串
            w_noise_distorted: [B, 1, H, W] 被攻击后的噪声图
        Returns:
            w_bits_pred: [B, num_bits] 解码的比特串
        """
        # 编码
        w_noise = self.encoder(w_bits)
        
        # 解码（从失真的噪声中）
        w_bits_pred = self.decoder(w_noise_distorted)
        
        return w_noise, w_bits_pred
    
    def encode(self, w_bits):
        """推理时：仅编码"""
        return self.encoder(w_bits)
    
    def decode(self, w_noise):
        """推理时：仅解码"""
        return self.decoder(w_noise)


class Stage1Loss(nn.Module):
    """
    Stage I 损失函数
    """
    def __init__(self, loss_bits_weight=1.0, loss_noise_weight=0.1):
        super().__init__()
        self.loss_bits_weight = loss_bits_weight
        self.loss_noise_weight = loss_noise_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, w_bits, w_bits_pred, w_noise, w_noise_distorted):
        """
        Args:
            w_bits: [B, num_bits] 原始比特串
            w_bits_pred: [B, num_bits] 解码的比特串
            w_noise: [B, 1, H, W] 编码的噪声
            w_noise_distorted: [B, 1, H, W] 失真后的噪声
        """
        # 比特准确性损失（主要目标）
        loss_bits = self.bce_loss(w_bits_pred, w_bits)
        
        # 噪声重建损失（辅助，确保编码质量）
        loss_noise = self.mse_loss(w_noise, w_noise_distorted)
        
        # 总损失
        total_loss = (self.loss_bits_weight * loss_bits + 
                     self.loss_noise_weight * loss_noise)
        
        return {
            'total': total_loss,
            'bits': loss_bits,
            'noise': loss_noise
        }
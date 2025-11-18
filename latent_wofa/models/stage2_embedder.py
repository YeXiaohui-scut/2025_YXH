"""
Stage II: 潜空间水印嵌入器
功能：将像素噪声"翻译"到VAE潜空间，并嵌入到生成图像中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseToLatentProjector(nn.Module):
    """
    将像素空间噪声图投影到VAE潜空间
    Input: w_noise [B, 1, 256, 256]
    Output: w_latent [B, 4, 64, 64]
    """
    def __init__(self, noise_size=256, latent_size=64, latent_channels=4):
        super().__init__()
        
        # 卷积下采样: 256 -> 64
        self.conv_layers = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 64 (保持尺寸，增加通道)
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 归一化到 [-1, 1]
        )
        
    def forward(self, w_noise):
        """
        Args:
            w_noise: [B, 1, 256, 256]
        Returns:
            w_latent: [B, 4, 64, 64]
        """
        return self.conv_layers(w_noise)


class AttentionFusion(nn.Module):
    """
    使用注意力机制融合原始潜码和水印潜码
    避免简单相加导致的图像质量下降
    """
    def __init__(self, channels=4):
        super().__init__()
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def forward(self, z_original, w_latent):
        """
        Args:
            z_original: [B, 4, 64, 64] 原始VAE潜码
            w_latent: [B, 4, 64, 64] 水印潜码
        Returns:
            z_watermarked: [B, 4, 64, 64] 带水印的潜码
        """
        # 拼接
        concat = torch.cat([z_original, w_latent], dim=1)  # [B, 8, 64, 64]
        
        # 空间注意力
        spatial_attn = self.spatial_attention(concat)  # [B, 1, 64, 64]
        
        # 通道注意力
        channel_attn = self.channel_attention(concat)  # [B, 4, 1, 1]
        
        # 融合
        fused = self.fusion_conv(concat)  # [B, 4, 64, 64]
        
        # 应用注意力权重
        fused = fused * spatial_attn * channel_attn
        
        # 残差连接（保持原始图像内容）
        z_watermarked = z_original + fused
        
        return z_watermarked


class LatentWatermarkEmbedder(nn.Module):
    """
    完整的潜空间水印嵌入器
    """
    def __init__(self, config):
        super().__init__()
        
        self.noise_to_latent = NoiseToLatentProjector(
            noise_size=config['watermark']['noise_size'],
            latent_size=config['stage2']['latent_size'],
            latent_channels=config['stage2']['latent_channels']
        )
        
        self.fusion_module = AttentionFusion(
            channels=config['stage2']['latent_channels']
        )
        
        # 可学习的嵌入强度参数
        self.watermark_strength = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, z_original, w_noise):
        """
        Args:
            z_original: [B, 4, 64, 64] 原始VAE潜码
            w_noise: [B, 1, 256, 256] 像素噪声图
        Returns:
            z_watermarked: [B, 4, 64, 64] 带水印的潜码
        """
        # 将像素噪声投影到潜空间
        w_latent = self.noise_to_latent(w_noise)  # [B, 4, 64, 64]
        
        # 缩放水印强度
        w_latent = w_latent * self.watermark_strength.abs()
        
        # 融合
        z_watermarked = self.fusion_module(z_original, w_latent)
        
        return z_watermarked


# ============ 测试代码 ============

if __name__ == "__main__":
    import yaml
    
    # 模拟配置
    config = {
        'watermark': {'noise_size': 256, 'num_bits': 48},
        'stage2': {
            'latent_size': 64,
            'latent_channels': 4
        }
    }
    
    # 测试嵌入器
    embedder = LatentWatermarkEmbedder(config)
    
    z_original = torch.randn(2, 4, 64, 64)  # 模拟VAE潜码
    w_noise = torch.randn(2, 1, 256, 256).tanh()  # 模拟像素噪声
    
    z_watermarked = embedder(z_original, w_noise)
    
    print(f"Original latent shape: {z_original.shape}")
    print(f"Watermarked latent shape: {z_watermarked.shape}")
    print(f"Watermark strength: {embedder.watermark_strength.item():.4f}")
    
    # 检查不可见性（潜码变化应该很小）
    diff = (z_watermarked - z_original).abs().mean()
    print(f"Average latent difference: {diff.item():.6f}")

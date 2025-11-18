"""
Stage II: 像素空间水印提取器
功能：从被攻击的像素图像中提取像素噪声图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    使用ResNet作为特征提取骨干网络
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        
        if backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 移除全连接层，保留卷积特征
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            features: list of [B, C_i, H_i, W_i] 多尺度特征
        """
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features


class DecoderBlock(nn.Module):
    """
    解码器上采样模块
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=4, stride=2, padding=1
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        """
        Args:
            x: [B, in_channels, H, W]
            skip: [B, skip_channels, 2H, 2W] 跳跃连接
        Returns:
            out: [B, out_channels, 2H, 2W]
        """
        x = self.upsample(x)
        
        if skip is not None:
            # 确保尺寸匹配
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class PixelWatermarkExtractor(nn.Module):
    """
    完整的像素空间水印提取器
    使用UNet架构，支持多尺度特征融合
    """
    def __init__(self, config):
        super().__init__()
        
        self.noise_size = config['watermark']['noise_size']
        backbone_name = config['stage2'].get('extractor_backbone', 'resnet50')
        
        # 编码器（骨干网络）
        self.encoder = ResNetBackbone(backbone_name, pretrained=True)
        
        # 解码器（上采样 + 跳跃连接）
        # ResNet50 channels: [256, 512, 1024, 2048]
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        # 最终上采样到原始分辨率
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()  # 输出 [-1, 1] 的噪声图
        )
        
    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] 被攻击的图像（H, W可变）
        Returns:
            w_noise_pred: [B, 1, noise_size, noise_size] 提取的噪声图
        """
        # 自适应调整输入尺寸
        original_size = image.shape[2:]
        if image.size(2) < 256 or image.size(3) < 256:
            image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 编码器提取多尺度特征
        features = self.encoder(image)  # [f1, f2, f3, f4]
        
        # 解码器逐步上采样 + 跳跃连接
        x = self.decoder4(features[3], features[2])  # 2048 -> 512
        x = self.decoder3(x, features[1])  # 512 -> 256
        x = self.decoder2(x, features[0])  # 256 -> 128
        x = self.decoder1(x, None)  # 128 -> 64
        
        # 最终上采样到目标尺寸
        w_noise_pred = self.final_upsample(x)  # [B, 1, H, W]
        
        # 调整到标准噪声尺寸
        if w_noise_pred.size(2) != self.noise_size or w_noise_pred.size(3) != self.noise_size:
            w_noise_pred = F.interpolate(
                w_noise_pred, 
                size=(self.noise_size, self.noise_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return w_noise_pred


# ============ 测试代码 ============

if __name__ == "__main__":
    config = {
        'watermark': {'noise_size': 256},
        'stage2': {'extractor_backbone': 'resnet50'}
    }
    
    extractor = PixelWatermarkExtractor(config)
    
    # 测试不同尺寸的输入
    test_images = [
        torch.randn(2, 3, 512, 512),  # 完整图像
        torch.randn(2, 3, 256, 256),  # 中等尺寸
        torch.randn(2, 3, 128, 128),  # 裁剪后的小图
    ]
    
    for i, img in enumerate(test_images):
        w_noise_pred = extractor(img)
        print(f"Input {i+1} shape: {img.shape} -> Output shape: {w_noise_pred.shape}")

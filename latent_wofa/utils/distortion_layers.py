"""
失真层模块
包含：
1. Stage I: 像素噪声的失真攻击（裁剪、旋转、高斯噪声）
2. Stage II: 像素图像的失真攻击（局部盗窃、生成式攻击、社交媒体攻击）
3. 渐进式课程学习控制器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
import cv2


class Stage1DistortionLayer(nn.Module):
    """
    Stage I 失真层：针对像素噪声图的攻击
    """
    def __init__(self, config, progressive_level='initial'):
        super().__init__()
        self.config = config['stage1']['distortion'][progressive_level]
        
    def random_crop(self, w_noise):
        """
        随机裁剪噪声图的一部分
        Args:
            w_noise: [B, 1, H, W]
        Returns:
            w_noise_cropped: [B, 1, H', W'] 被裁剪的噪声
        """
        B, C, H, W = w_noise.shape
        
        # 随机裁剪比例
        crop_ratio = random.uniform(
            self.config['crop_ratio_min'],
            self.config['crop_ratio_max']
        )
        
        # 计算裁剪尺寸
        new_h = int(H * np.sqrt(crop_ratio))
        new_w = int(W * np.sqrt(crop_ratio))
        new_h = max(new_h, 16)  # 最小尺寸
        new_w = max(new_w, 16)
        
        # 随机裁剪位置
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        
        w_noise_cropped = w_noise[:, :, top:top+new_h, left:left+new_w]
        
        return w_noise_cropped
    
    def random_rotation(self, w_noise):
        """
        随机旋转噪声图
        Args:
            w_noise: [B, 1, H, W]
        Returns:
            w_noise_rotated: [B, 1, H, W]
        """
        angle = random.uniform(
            -self.config['rotation_degrees'],
            self.config['rotation_degrees']
        )
        
        # 对batch中的每个样本单独旋转
        rotated_list = []
        for i in range(w_noise.size(0)):
            rotated = TF.rotate(
                w_noise[i:i+1], 
                angle, 
                interpolation=TF.InterpolationMode.BILINEAR
            )
            rotated_list.append(rotated)
        
        return torch.cat(rotated_list, dim=0)
    
    def add_gaussian_noise(self, w_noise):
        """
        添加高斯噪声
        Args:
            w_noise: [B, 1, H, W]
        Returns:
            w_noise_noisy: [B, 1, H, W]
        """
        noise = torch.randn_like(w_noise) * self.config['gaussian_noise_std']
        return w_noise + noise
    
    def forward(self, w_noise):
        """
        综合应用多种失真
        Args:
            w_noise: [B, 1, H, W] 编码的噪声图
        Returns:
            w_noise_distorted: [B, 1, H', W'] 失真后的噪声
        """
        # 随机选择应用哪些失真
        if random.random() > 0.3:
            w_noise = self.random_crop(w_noise)
        
        if random.random() > 0.3:
            w_noise = self.random_rotation(w_noise)
        
        if random.random() > 0.5:
            w_noise = self.add_gaussian_noise(w_noise)
        
        return w_noise


class CropAndFuseAttack(nn.Module):
    """
    局部盗窃攻击：裁剪 + 几何变换 + 背景融合
    模拟恶意用户"抠图"行为
    """
    def __init__(self, config, background_images=None):
        super().__init__()
        self.config = config['stage2']['distortion']['crop_and_fuse']
        self.background_images = background_images  # 背景图像库
        
    def random_crop_mask(self, image):
        """
        生成随机裁剪掩码（模拟抠图）
        Args:
            image: [B, 3, H, W]
        Returns:
            mask: [B, 1, H, W] binary mask
            cropped_image: [B, 3, H, W] 裁剪后的图像（其他区域为0）
        """
        B, C, H, W = image.shape
        device = image.device
        
        crop_ratio = random.uniform(
            self.config['crop_ratio_min'],
            self.config['crop_ratio_max']
        )
        
        # 生成不规则形状的掩码
        mask = torch.zeros(B, 1, H, W, device=device)
        
        for i in range(B):
            # 随机选择形状：矩形、圆形、不规则
            shape_type = random.choice(['rectangle', 'ellipse', 'irregular'])
            
            if shape_type == 'rectangle':
                # 矩形裁剪
                crop_h = int(H * np.sqrt(crop_ratio))
                crop_w = int(W * np.sqrt(crop_ratio))
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                mask[i, 0, top:top+crop_h, left:left+crop_w] = 1.0
                
            elif shape_type == 'ellipse':
                # 椭圆形裁剪
                center_y = random.randint(H // 4, 3 * H // 4)
                center_x = random.randint(W // 4, 3 * W // 4)
                radius_y = int(H * np.sqrt(crop_ratio) / 2)
                radius_x = int(W * np.sqrt(crop_ratio) / 2)
                
                y, x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                ellipse_mask = (
                    ((y - center_y) / radius_y) ** 2 + 
                    ((x - center_x) / radius_x) ** 2
                ) <= 1.0
                mask[i, 0] = ellipse_mask.float()
            
            else:
                # 不规则形状（随机多边形）
                num_points = random.randint(5, 10)
                points = np.random.rand(num_points, 2)
                points[:, 0] *= W
                points[:, 1] *= H
                
                # 使用OpenCV生成多边形掩码
                mask_np = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask_np, [points.astype(np.int32)], 1)
                mask[i, 0] = torch.from_numpy(mask_np).float().to(device)
        
        cropped_image = image * mask
        return mask, cropped_image
    
    def geometric_transform(self, image, mask):
        """
        应用几何变换（旋转、缩放）
        Args:
            image: [B, 3, H, W]
            mask: [B, 1, H, W]
        Returns:
            transformed_image: [B, 3, H, W]
            transformed_mask: [B, 1, H, W]
        """
        # 旋转
        angle = random.uniform(
            -self.config['rotation_degrees'],
            self.config['rotation_degrees']
        )
        
        # 缩放
        scale = random.uniform(
            self.config['scale_range'][0],
            self.config['scale_range'][1]
        )
        
        transformed_images = []
        transformed_masks = []
        
        for i in range(image.size(0)):
            # 旋转
            img_rot = TF.rotate(
                image[i:i+1], 
                angle,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            mask_rot = TF.rotate(
                mask[i:i+1],
                angle,
                interpolation=TF.InterpolationMode.NEAREST
            )
            
            # 缩放
            H, W = img_rot.shape[2:]
            new_H, new_W = int(H * scale), int(W * scale)
            
            img_scaled = F.interpolate(
                img_rot,
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            )
            mask_scaled = F.interpolate(
                mask_rot,
                size=(new_H, new_W),
                mode='nearest'
            )
            
            # 调整回原始尺寸（裁剪或padding）
            if scale > 1.0:
                # 裁剪
                top = (new_H - H) // 2
                left = (new_W - W) // 2
                img_scaled = img_scaled[:, :, top:top+H, left:left+W]
                mask_scaled = mask_scaled[:, :, top:top+H, left:left+W]
            else:
                # Padding
                pad_h = (H - new_H) // 2
                pad_w = (W - new_W) // 2
                img_scaled = F.pad(
                    img_scaled,
                    (pad_w, W - new_W - pad_w, pad_h, H - new_H - pad_h)
                )
                mask_scaled = F.pad(
                    mask_scaled,
                    (pad_w, W - new_W - pad_w, pad_h, H - new_H - pad_h)
                )
            
            transformed_images.append(img_scaled)
            transformed_masks.append(mask_scaled)
        
        return torch.cat(transformed_images, dim=0), torch.cat(transformed_masks, dim=0)
    
    def fuse_with_background(self, cropped_image, mask):
        """
        将裁剪的图像融合到新背景
        Args:
            cropped_image: [B, 3, H, W]
            mask: [B, 1, H, W]
        Returns:
            fused_image: [B, 3, H, W]
        """
        B, C, H, W = cropped_image.shape
        device = cropped_image.device
        
        if self.background_images is not None and len(self.background_images) > 0:
            # 从背景库中随机选择
            backgrounds = []
            for _ in range(B):
                bg_idx = random.randint(0, len(self.background_images) - 1)
                bg = self.background_images[bg_idx].to(device)
                bg = F.interpolate(
                    bg.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                backgrounds.append(bg)
            background = torch.cat(backgrounds, dim=0)
        else:
            # 生成随机噪声背景
            background = torch.randn(B, C, H, W, device=device) * 0.5
        
        # 融合
        fused_image = cropped_image * mask + background * (1 - mask)
        
        return fused_image
    
    def forward(self, image):
        """
        完整的局部盗窃攻击流程
        Args:
            image: [B, 3, H, W] 水印图像
        Returns:
            attacked_image: [B, 3, H, W] 被攻击后的图像
        """
        # 1. 裁剪（抠图）
        mask, cropped_image = self.random_crop_mask(image)
        
        # 2. 几何变换
        transformed_image, transformed_mask = self.geometric_transform(
            cropped_image, mask
        )
        
        # 3. 背景融合
        fused_image = self.fuse_with_background(
            transformed_image, transformed_mask
        )
        
        return fused_image


class JPEGCompression(nn.Module):
    """
    JPEG压缩攻击
    """
    def __init__(self, quality_range=(30, 90)):
        super().__init__()
        self.quality_range = quality_range
        
    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] tensor in [-1, 1]
        Returns:
            compressed: [B, 3, H, W]
        """
        # 注意：这是简化实现，实际应使用DiffJPEG等可微JPEG库
        # 这里仅做演示
        return image + torch.randn_like(image) * 0.02  # 模拟压缩噪声


class Stage2DistortionLayer(nn.Module):
    """
    Stage II 完整失真层
    """
    def __init__(self, config, progressive_level='initial', background_images=None):
        super().__init__()
        self.config = config
        self.progressive_level = progressive_level
        
        # 各种攻击模块
        self.crop_and_fuse = CropAndFuseAttack(config, background_images)
        self.jpeg_compression = JPEGCompression(
            quality_range=config['stage2']['distortion']['social_media']['jpeg_quality_range']
        )
        
    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] 水印图像
        Returns:
            attacked_image: [B, 3, H, W]
        """
        # 主要攻击：局部盗窃
        image = self.crop_and_fuse(image)
        
        # 可选攻击
        if random.random() > 0.5:
            image = self.jpeg_compression(image)
        
        return image


# ============ 测试代码 ============

if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 测试 Stage I 失真层
    print("Testing Stage I Distortion Layer...")
    stage1_distortion = Stage1DistortionLayer(config, progressive_level='final')
    w_noise = torch.randn(4, 1, 256, 256).tanh()
    w_noise_distorted = stage1_distortion(w_noise)
    print(f"Input shape: {w_noise.shape}")
    print(f"Output shape: {w_noise_distorted.shape}")
    
    # 测试 Stage II 失真层
    print("\nTesting Stage II Distortion Layer...")
    stage2_distortion = Stage2DistortionLayer(config, progressive_level='final')
    image = torch.randn(4, 3, 512, 512).tanh()
    attacked_image = stage2_distortion(image)
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {attacked_image.shape}")

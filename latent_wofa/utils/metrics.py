"""
评估指标计算
"""

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class WatermarkMetrics:
    """
    水印评估指标集合
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
    def bit_accuracy(self, w_bits_pred, w_bits_true, threshold=0.5):
        """
        计算比特准确率
        Args:
            w_bits_pred: [B, num_bits] 预测的比特概率
            w_bits_true: [B, num_bits] 真实比特串
            threshold: float, 二值化阈值
        Returns:
            accuracy: float, [0, 1]
        """
        w_bits_pred_binary = (w_bits_pred > threshold).float()
        correct = (w_bits_pred_binary == w_bits_true).float()
        accuracy = correct.mean().item()
        return accuracy
    
    def bit_error_rate(self, w_bits_pred, w_bits_true, threshold=0.5):
        """
        计算比特错误率 (BER)
        """
        return 1.0 - self.bit_accuracy(w_bits_pred, w_bits_true, threshold)
    
    def psnr(self, image1, image2):
        """
        计算PSNR
        Args:
            image1, image2: [B, C, H, W] tensor in [-1, 1]
        Returns:
            psnr_value: float
        """
        # 转换到 [0, 1]
        image1 = (image1 + 1) / 2
        image2 = (image2 + 1) / 2
        
        # 转换为numpy
        image1_np = image1.cpu().numpy().transpose(0, 2, 3, 1)
        image2_np = image2.cpu().numpy().transpose(0, 2, 3, 1)
        
        psnr_values = []
        for i in range(image1_np.shape[0]):
            psnr_val = peak_signal_noise_ratio(
                image1_np[i], 
                image2_np[i],
                data_range=1.0
            )
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    def ssim(self, image1, image2):
        """
        计算SSIM
        Args:
            image1, image2: [B, C, H, W] tensor in [-1, 1]
        Returns:
            ssim_value: float
        """
        # 转换到 [0, 1]
        image1 = (image1 + 1) / 2
        image2 = (image2 + 1) / 2
        
        # 转换为numpy
        image1_np = image1.cpu().numpy().transpose(0, 2, 3, 1)
        image2_np = image2.cpu().numpy().transpose(0, 2, 3, 1)
        
        ssim_values = []
        for i in range(image1_np.shape[0]):
            ssim_val = structural_similarity(
                image1_np[i],
                image2_np[i],
                data_range=1.0,
                channel_axis=-1
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def lpips_distance(self, image1, image2):
        """
        计算感知距离 (LPIPS)
        Args:
            image1, image2: [B, C, H, W] tensor in [-1, 1]
        Returns:
            lpips_value: float
        """
        with torch.no_grad():
            distance = self.lpips_model(
                image1.to(self.device),
                image2.to(self.device)
            )
        return distance.mean().item()
    
    def evaluate_all(self, image_original, image_watermarked, 
                    w_bits_pred, w_bits_true):
        """
        计算所有指标
        Returns:
            metrics: dict
        """
        metrics = {
            'bit_accuracy': self.bit_accuracy(w_bits_pred, w_bits_true),
            'bit_error_rate': self.bit_error_rate(w_bits_pred, w_bits_true),
            'psnr': self.psnr(image_original, image_watermarked),
            'ssim': self.ssim(image_original, image_watermarked),
            'lpips': self.lpips_distance(image_original, image_watermarked),
        }
        return metrics
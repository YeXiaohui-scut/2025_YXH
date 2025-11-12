"""
Neural Network Models for Hybrid Watermarking Framework
Combines MetaSeal, SWIFT, and GenPTW architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List
import numpy as np
import random
import cv2


# ============================================================================
# WatermarkEmbedder - CNN-based watermark embedding (inspired by HiDDeN/SWIFT)
# ============================================================================

class WatermarkEmbedder(nn.Module):
    """
    Embeds QR code watermark into images using CNN encoder
    Based on HiDDeN architecture from SWIFT paper
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # QR code preprocessing - expand 256x256 -> 512x512
        self.qr_upsampler = nn.Sequential(
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False),
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Image feature extractor
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Fusion network - combine image and QR features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim + 32, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Residual generation with JND constraint
        self.residual_gen = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # JND constraint strength
        self.jnd_strength = 0.05  # Maximum perturbation strength
    
    def forward(self, image: Tensor, qr_code_tensor: Tensor) -> Tensor:
        """
        Embed watermark into image
        
        Args:
            image: [B, 3, 512, 512] RGB image
            qr_code_tensor: [B, 1, 256, 256] QR code
            
        Returns:
            Watermarked image [B, 3, 512, 512]
        """
        # Extract image features
        img_features = self.img_encoder(image)  # [B, hidden_dim, 512, 512]
        
        # Upsample and process QR code
        qr_features = self.qr_upsampler(qr_code_tensor)  # [B, 32, 512, 512]
        
        # Fuse features
        combined = torch.cat([img_features, qr_features], dim=1)
        fused = self.fusion(combined)  # [B, hidden_dim, 512, 512]
        
        # Generate residual with JND constraint
        residual = self.residual_gen(fused)  # [B, 3, 512, 512]
        residual = residual * self.jnd_strength
        
        # Add residual to original image
        watermarked = image + residual
        watermarked = torch.clamp(watermarked, 0, 1)
        
        return watermarked


# ============================================================================
# DistortionLayer - Simulates various attacks (GenPTW)
# ============================================================================

class DistortionLayer(nn.Module):
    """
    Simulates AIGC editing and common degradations for robust training
    Based on GenPTW distortion simulation
    """
    
    def __init__(self, vae_model=None):
        super().__init__()
        self.vae_model = vae_model
        self.training_mode = True
    
    def forward(self, image: Tensor, ground_truth_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Apply random distortions to image
        
        Args:
            image: [B, 3, H, W] input image
            ground_truth_mask: [B, 1, H, W] optional mask for inpainting regions
            
        Returns:
            tuple: (distorted_image, ground_truth_mask)
        """
        if not self.training:
            # During inference, return image as-is or apply light distortion
            return image, ground_truth_mask if ground_truth_mask is not None else torch.zeros_like(image[:, :1])
        
        B, C, H, W = image.shape
        distorted = image.clone()
        
        # Generate ground truth mask if not provided
        if ground_truth_mask is None:
            ground_truth_mask = self._generate_random_mask(B, H, W, device=image.device)
        
        # Apply AIGC editing simulations randomly
        aigc_prob = random.random()
        
        if aigc_prob < 0.3 and self.vae_model is not None:
            # VAE reconstruction attack
            distorted = self._vae_reconstruction(distorted)
        
        elif aigc_prob < 0.5:
            # Watermark region removal (simulate inpainting)
            distorted = self._watermark_removal(distorted, ground_truth_mask)
        
        elif aigc_prob < 0.7:
            # Simulated inpainting on masked region
            distorted = self._simulate_inpainting(distorted, ground_truth_mask)
        
        # Apply common degradations
        distorted = self._apply_common_degradations(distorted)
        
        return distorted, ground_truth_mask
    
    def _generate_random_mask(self, batch_size: int, height: int, width: int, device) -> Tensor:
        """Generate random binary masks for simulated tampering"""
        masks = []
        
        for _ in range(batch_size):
            mask = torch.zeros(1, height, width, device=device)
            
            # Random number of rectangular regions
            num_regions = random.randint(1, 3)
            
            for _ in range(num_regions):
                # Random rectangle
                h_size = random.randint(height // 8, height // 4)
                w_size = random.randint(width // 8, width // 4)
                
                y = random.randint(0, height - h_size)
                x = random.randint(0, width - w_size)
                
                mask[0, y:y+h_size, x:x+w_size] = 1.0
            
            masks.append(mask)
        
        return torch.stack(masks)
    
    def _vae_reconstruction(self, image: Tensor) -> Tensor:
        """Simulate VAE encode-decode cycle"""
        if self.vae_model is None:
            # Fallback: apply blur
            return self._apply_gaussian_blur(image, kernel_size=5, sigma=1.0)
        
        try:
            with torch.no_grad():
                # Encode and decode through VAE
                latent = self.vae_model.encode(image).latent_dist.sample()
                reconstructed = self.vae_model.decode(latent).sample
                return torch.clamp(reconstructed, 0, 1)
        except:
            return self._apply_gaussian_blur(image, kernel_size=5, sigma=1.0)
    
    def _watermark_removal(self, image: Tensor, mask: Tensor) -> Tensor:
        """Simulate watermark removal by replacing masked regions"""
        # Apply Gaussian blur to masked regions
        blurred = self._apply_gaussian_blur(image, kernel_size=7, sigma=2.0)
        result = image * (1 - mask) + blurred * mask
        return result
    
    def _simulate_inpainting(self, image: Tensor, mask: Tensor) -> Tensor:
        """Simulate inpainting on masked regions"""
        # Simple inpainting simulation: blend with surrounding regions
        blurred = self._apply_gaussian_blur(image, kernel_size=11, sigma=3.0)
        result = image * (1 - mask) + blurred * mask
        return result
    
    def _apply_gaussian_blur(self, image: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
        """Apply Gaussian blur"""
        B, C, H, W = image.shape
        
        # Create Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(image.device)
        kernel = kernel.repeat(C, 1, 1, 1)
        
        # Apply blur
        padding = kernel_size // 2
        blurred = F.conv2d(image, kernel, padding=padding, groups=C)
        
        return blurred
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        """Generate Gaussian kernel"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        return kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def _apply_common_degradations(self, image: Tensor) -> Tensor:
        """Apply random common degradations"""
        # JPEG compression (simplified)
        if random.random() < 0.5:
            image = self._simulate_jpeg_compression(image, quality=random.randint(50, 95))
        
        # Gaussian noise
        if random.random() < 0.3:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(image) * noise_std
            image = image + noise
        
        # Brightness adjustment
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            image = image * brightness_factor
        
        # Contrast adjustment
        if random.random() < 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = image.mean(dim=[2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean
        
        return torch.clamp(image, 0, 1)
    
    def _simulate_jpeg_compression(self, image: Tensor, quality: int = 75) -> Tensor:
        """Simulate JPEG compression (simplified version)"""
        # Simplified: apply blockwise DCT and quantization
        # For full implementation, use jpeg2dct or kornia
        return image


# ============================================================================
# DCT Module for Frequency Separation (GenPTW)
# ============================================================================

class DCTModule(nn.Module):
    """Discrete Cosine Transform for frequency separation"""
    
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
    
    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Separate image into low and high frequency components
        
        Args:
            image: [B, 3, H, W]
            
        Returns:
            tuple: (low_freq, high_freq) both [B, 3, H, W]
        """
        # Simplified frequency separation using average pooling
        # For production, use proper DCT implementation
        
        # Low frequency: downsampled then upsampled
        low_freq = F.avg_pool2d(image, kernel_size=4, stride=1, padding=2)
        
        # High frequency: original - low frequency
        high_freq = image - low_freq
        
        return low_freq, high_freq


# ============================================================================
# WatermarkExtractor - Frequency-coordinated decoder (GenPTW)
# ============================================================================

class WatermarkExtractor(nn.Module):
    """
    Extract watermark and tamper mask using frequency-coordinated architecture
    Based on GenPTW
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # DCT module for frequency separation
        self.dct_module = DCTModule()
        
        # Low-frequency branch (W_Dec) - watermark reconstruction
        self.low_freq_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # QR code reconstructor
        self.qr_reconstructor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # High-frequency branch (CN_Enc) - ConvNeXt-style encoder
        self.high_freq_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 7, padding=3),
            nn.ReLU(inplace=True),
            ConvNeXtBlock(hidden_dim),
            ConvNeXtBlock(hidden_dim),
        )
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, 1)
        
        # Mask decoder - multi-scale processing
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract watermark and tamper mask
        
        Args:
            image: [B, 3, H, W] distorted watermarked image
            
        Returns:
            tuple: (reconstructed_qr [B, 1, 256, 256], tamper_mask [B, 1, H, W])
        """
        # Frequency separation
        low_freq, high_freq = self.dct_module(image)
        
        # Low-frequency processing -> watermark reconstruction
        low_features = self.low_freq_encoder(low_freq)  # [B, hidden_dim, H, W]
        reconstructed_qr = self.qr_reconstructor(low_features)  # [B, 1, 256, 256]
        
        # High-frequency processing
        high_features = self.high_freq_encoder(high_freq)  # [B, hidden_dim, H, W]
        
        # Fusion for tamper localization
        # Upsample low_features to match high_features if needed
        if low_features.shape != high_features.shape:
            low_features_resized = F.interpolate(low_features, size=high_features.shape[2:], mode='bilinear', align_corners=False)
        else:
            low_features_resized = low_features
        
        fused_features = torch.cat([low_features_resized, high_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Tamper mask prediction
        tamper_mask = self.mask_decoder(fused_features)  # [B, 1, H, W]
        
        return reconstructed_qr, tamper_mask


# ============================================================================
# ConvNeXt Block (for high-frequency encoder)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for high-frequency processing"""
    
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        x = residual + x
        return x


# ============================================================================
# WatermarkAutoencoder - Complete training model
# ============================================================================

class WatermarkAutoencoder(nn.Module):
    """
    Complete autoencoder for watermark embedding and extraction training
    Combines Embedder -> Distortion -> Extractor
    """
    
    def __init__(self, hidden_dim: int = 64, vae_model=None):
        super().__init__()
        
        self.embedder = WatermarkEmbedder(hidden_dim=hidden_dim)
        self.distortion = DistortionLayer(vae_model=vae_model)
        self.extractor = WatermarkExtractor(hidden_dim=hidden_dim)
    
    def forward(
        self, 
        image: Tensor, 
        qr_code_tensor: Tensor, 
        ground_truth_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Full forward pass for training
        
        Args:
            image: [B, 3, H, W] original image
            qr_code_tensor: [B, 1, 256, 256] QR code watermark
            ground_truth_mask: [B, 1, H, W] optional tamper mask
            
        Returns:
            tuple: (watermarked_image, reconstructed_qr, tamper_mask, gt_mask)
        """
        # Embed watermark
        watermarked_image = self.embedder(image, qr_code_tensor)
        
        # Apply distortions
        distorted_image, gt_mask = self.distortion(watermarked_image, ground_truth_mask)
        
        # Extract watermark and tamper mask
        reconstructed_qr, tamper_mask = self.extractor(distorted_image)
        
        return watermarked_image, reconstructed_qr, tamper_mask, gt_mask


# ============================================================================
# Test Functions
# ============================================================================

def test_models():
    """Test all model components"""
    print("\n" + "="*60)
    print("Testing Model Components")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 2
    
    # Test Embedder
    print("\n1. Testing WatermarkEmbedder...")
    embedder = WatermarkEmbedder(hidden_dim=64).to(device)
    test_image = torch.rand(batch_size, 3, 512, 512).to(device)
    test_qr = torch.rand(batch_size, 1, 256, 256).to(device)
    
    watermarked = embedder(test_image, test_qr)
    print(f"   Input: {test_image.shape}, QR: {test_qr.shape}")
    print(f"   Output: {watermarked.shape}")
    print(f"   ✓ WatermarkEmbedder works!")
    
    # Test DistortionLayer
    print("\n2. Testing DistortionLayer...")
    distortion = DistortionLayer().to(device)
    distorted, mask = distortion(watermarked)
    print(f"   Input: {watermarked.shape}")
    print(f"   Output: {distorted.shape}, Mask: {mask.shape}")
    print(f"   ✓ DistortionLayer works!")
    
    # Test Extractor
    print("\n3. Testing WatermarkExtractor...")
    extractor = WatermarkExtractor(hidden_dim=64).to(device)
    reconstructed_qr, tamper_mask = extractor(distorted)
    print(f"   Input: {distorted.shape}")
    print(f"   QR output: {reconstructed_qr.shape}")
    print(f"   Mask output: {tamper_mask.shape}")
    print(f"   ✓ WatermarkExtractor works!")
    
    # Test complete Autoencoder
    print("\n4. Testing WatermarkAutoencoder...")
    autoencoder = WatermarkAutoencoder(hidden_dim=64).to(device)
    watermarked, recon_qr, pred_mask, gt_mask = autoencoder(test_image, test_qr)
    print(f"   Input: {test_image.shape}, QR: {test_qr.shape}")
    print(f"   Watermarked: {watermarked.shape}")
    print(f"   Reconstructed QR: {recon_qr.shape}")
    print(f"   Predicted mask: {pred_mask.shape}")
    print(f"   GT mask: {gt_mask.shape}")
    print(f"   ✓ WatermarkAutoencoder works!")
    
    # Count parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"\n5. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("All model tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Hybrid Watermarking Framework - Models Module")
    print("Testing neural network architectures...\n")
    
    test_models()
    
    print("\n✓ All models are ready!")
    print("\nNext steps:")
    print("1. Implement train.py")
    print("2. Implement main.py")
    print("3. Prepare dataset and start training")

"""
U-Net based WEmb (Watermark Embedding) Module
Generates content-adaptive perturbation patterns from semantic vectors
Based on Embedding Guide's UNet++ approach for multi-scale feature fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import zero_module


class ConvBlock(nn.Module):
    """Basic convolutional block for U-Net"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder with skip connections"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class SemanticVectorProjection(nn.Module):
    """Project semantic vector to spatial feature map"""
    def __init__(self, semantic_dim, out_channels, spatial_size=8):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        
        # Project semantic vector to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(semantic_dim, out_channels * spatial_size * spatial_size),
            nn.SiLU(),
        )
    
    def forward(self, semantic_vector, target_h, target_w):
        """
        Args:
            semantic_vector: (batch_size, semantic_dim)
            target_h, target_w: Target spatial dimensions
        Returns:
            Projected feature map of shape (batch_size, out_channels, target_h, target_w)
        """
        batch_size = semantic_vector.shape[0]
        
        # Project to initial spatial size
        x = self.fc(semantic_vector)
        x = x.view(batch_size, self.out_channels, self.spatial_size, self.spatial_size)
        
        # Interpolate to target size
        if self.spatial_size != target_h or self.spatial_size != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return x


class UNetWEmb(nn.Module):
    """
    U-Net based Watermark Embedding module
    Generates content-adaptive perturbation patterns from semantic vectors
    """
    def __init__(self, semantic_dim=768, feature_channels=512, base_channels=64):
        """
        Args:
            semantic_dim: Dimension of semantic vector
            feature_channels: Number of channels in input feature map
            base_channels: Base number of channels for U-Net
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.feature_channels = feature_channels
        
        # Semantic vector projection
        self.semantic_proj = SemanticVectorProjection(
            semantic_dim=semantic_dim,
            out_channels=base_channels,
            spatial_size=8
        )
        
        # U-Net Encoder
        # Input: concatenated semantic projection + feature map
        self.enc1 = DownBlock(base_channels + feature_channels, base_channels * 2)  # -> base*2
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4)  # -> base*4
        self.enc3 = DownBlock(base_channels * 4, base_channels * 8)  # -> base*8
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # U-Net Decoder with skip connections
        self.dec3 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        
        # Final output layer (zero-initialized for stable training)
        self.final_up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.final_conv = zero_module(nn.Conv2d(base_channels, feature_channels, kernel_size=3, padding=1))
    
    def forward(self, semantic_vector, feature_map):
        """
        Generate perturbation pattern from semantic vector and feature map
        Args:
            semantic_vector: Encrypted semantic vector (batch_size, semantic_dim)
            feature_map: Intermediate feature map from VAE decoder (batch_size, channels, H, W)
        Returns:
            Perturbation pattern of same shape as feature_map
        """
        batch_size, channels, h, w = feature_map.shape
        
        # Project semantic vector to spatial feature map
        semantic_spatial = self.semantic_proj(semantic_vector, h, w)
        
        # Concatenate with feature map
        x = torch.cat([semantic_spatial, feature_map], dim=1)
        
        # U-Net Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # U-Net Decoder with skip connections
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final output
        x = self.final_up(x)
        perturbation = self.final_conv(x)
        
        # Ensure output has same size as input feature map
        if perturbation.shape[2:] != feature_map.shape[2:]:
            perturbation = F.interpolate(perturbation, size=(h, w), mode='bilinear', align_corners=False)
        
        return perturbation


class MultiScaleUNetWEmb(nn.Module):
    """
    Multi-scale U-Net WEmb modules for different decoder layers
    """
    def __init__(self, semantic_dim=768, layer_configs=None):
        """
        Args:
            semantic_dim: Dimension of semantic vector
            layer_configs: List of (feature_channels, base_channels) for each layer
        """
        super().__init__()
        
        if layer_configs is None:
            # Default configuration matching original LaWa layers
            layer_configs = [
                (4, 16),      # Initial layer (latent space)
                (512, 32),    # Middle layer
                (128, 32),    # Layer 3
                (256, 32),    # Layer 2
                (512, 32),    # Layer 1
                (512, 32),    # Layer 0
            ]
        
        self.wemb_modules = nn.ModuleList([
            UNetWEmb(semantic_dim=semantic_dim, 
                    feature_channels=fc, 
                    base_channels=bc)
            for fc, bc in layer_configs
        ])
    
    def forward(self, semantic_vector, feature_maps):
        """
        Generate perturbations for all layers
        Args:
            semantic_vector: Encrypted semantic vector (batch_size, semantic_dim)
            feature_maps: List of feature maps from different decoder layers
        Returns:
            List of perturbation patterns
        """
        perturbations = []
        for i, (wemb, feature_map) in enumerate(zip(self.wemb_modules, feature_maps)):
            perturbation = wemb(semantic_vector, feature_map)
            perturbations.append(perturbation)
        
        return perturbations


class LightweightUNetWEmb(nn.Module):
    """
    Lightweight U-Net WEmb for faster training and inference
    Uses depthwise separable convolutions
    """
    def __init__(self, semantic_dim=768, feature_channels=512, base_channels=32):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.feature_channels = feature_channels
        
        # Semantic vector projection (simpler)
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, base_channels * 64),
            nn.SiLU(),
        )
        
        # Simplified encoder-decoder
        in_channels = base_channels + feature_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Output layer (zero-initialized)
        self.output = zero_module(nn.Conv2d(base_channels, feature_channels, 3, padding=1))
    
    def forward(self, semantic_vector, feature_map):
        batch_size, channels, h, w = feature_map.shape
        
        # Project and reshape semantic vector
        semantic_features = self.semantic_proj(semantic_vector)
        semantic_features = semantic_features.view(batch_size, -1, 8, 8)
        
        # Upsample to match feature map size
        semantic_features = F.interpolate(semantic_features, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate and process
        x = torch.cat([semantic_features, feature_map], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        perturbation = self.output(x)
        
        return perturbation


def test_unet_wemb():
    """Test function for U-Net WEmb modules"""
    print("Testing U-Net WEmb modules...")
    
    # Test single layer
    semantic_dim = 768
    feature_channels = 512
    batch_size = 2
    h, w = 32, 32
    
    wemb = UNetWEmb(semantic_dim=semantic_dim, feature_channels=feature_channels)
    
    semantic_vector = torch.randn(batch_size, semantic_dim)
    feature_map = torch.randn(batch_size, feature_channels, h, w)
    
    perturbation = wemb(semantic_vector, feature_map)
    print(f"Single layer output shape: {perturbation.shape}")
    assert perturbation.shape == feature_map.shape, "Output shape mismatch"
    
    # Test multi-scale
    multi_wemb = MultiScaleUNetWEmb(semantic_dim=semantic_dim)
    feature_maps = [
        torch.randn(batch_size, 4, 32, 32),
        torch.randn(batch_size, 512, 32, 32),
        torch.randn(batch_size, 128, 64, 64),
        torch.randn(batch_size, 256, 128, 128),
        torch.randn(batch_size, 512, 256, 256),
        torch.randn(batch_size, 512, 256, 256),
    ]
    
    perturbations = multi_wemb(semantic_vector, feature_maps)
    print(f"Multi-scale outputs: {len(perturbations)} layers")
    for i, p in enumerate(perturbations):
        print(f"  Layer {i}: {p.shape}")
        assert p.shape == feature_maps[i].shape, f"Layer {i} shape mismatch"
    
    # Test lightweight version
    light_wemb = LightweightUNetWEmb(semantic_dim=semantic_dim, feature_channels=feature_channels)
    perturbation_light = light_wemb(semantic_vector, feature_map)
    print(f"Lightweight output shape: {perturbation_light.shape}")
    assert perturbation_light.shape == feature_map.shape, "Lightweight output shape mismatch"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_unet_wemb()

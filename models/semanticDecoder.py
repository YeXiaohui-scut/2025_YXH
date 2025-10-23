"""
Semantic Watermark Decoder
Extracts semantic vectors from watermarked images
"""
import torch
import torch.nn as nn
import torchvision


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class SemanticDecoder(nn.Module):
    """
    Decoder that extracts semantic vectors from watermarked images
    Uses ResNet backbone followed by projection to semantic space
    """
    def __init__(self, semantic_dim=768, pretrained_weights=None):
        """
        Args:
            semantic_dim: Dimension of semantic vector to extract
            pretrained_weights: Path to pretrained weights
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        
        # Use ResNet50 as backbone
        if pretrained_weights is not None:
            self.backbone = torchvision.models.resnet50(pretrained=False, progress=False)
            checkpoint = torch.load(pretrained_weights)
            self.backbone.load_state_dict(checkpoint)
        else:
            self.backbone = torchvision.models.resnet50(pretrained=True, progress=False)
        
        # Replace final FC layer with semantic vector projection
        backbone_features = self.backbone.fc.in_features
        
        # Multi-layer projection for better feature learning
        self.projection = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, semantic_dim)
        )
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
    
    def forward(self, image):
        """
        Extract semantic vector from image
        Args:
            image: Input image tensor (batch_size, 3, H, W)
        Returns:
            Extracted semantic vector (batch_size, semantic_dim)
        """
        # Extract features using backbone
        features = self.backbone(image)
        
        # Project to semantic space
        semantic_vector = self.projection(features)
        
        return semantic_vector


class LightweightSemanticDecoder(nn.Module):
    """
    Lightweight semantic decoder using MobileNetV2
    """
    def __init__(self, semantic_dim=768):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        
        # Use MobileNetV2 as backbone (lighter)
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
        
        # Get feature dimension
        backbone_features = self.backbone.last_channel
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_features, semantic_dim)
        )
    
    def forward(self, image):
        return self.backbone(image)


class MultiScaleSemanticDecoder(nn.Module):
    """
    Multi-scale semantic decoder that extracts features at multiple scales
    Useful for robust watermark extraction
    """
    def __init__(self, semantic_dim=768):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        
        # Use ResNet50 as backbone
        backbone = torchvision.models.resnet50(pretrained=True, progress=False)
        
        # Extract intermediate layers
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, semantic_dim)
        )
    
    def forward(self, image):
        # Extract multi-scale features
        x1 = self.layer1(image)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Global average pooling
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        
        # Project to semantic space
        semantic_vector = self.fusion(x)
        
        return semantic_vector


def test_semantic_decoder():
    """Test function for semantic decoder"""
    print("Testing Semantic Decoder...")
    
    semantic_dim = 768
    batch_size = 2
    
    # Test standard decoder
    decoder = SemanticDecoder(semantic_dim=semantic_dim)
    image = torch.randn(batch_size, 3, 256, 256)
    output = decoder(image)
    print(f"Standard decoder output shape: {output.shape}")
    assert output.shape == (batch_size, semantic_dim), "Output shape mismatch"
    
    # Test lightweight decoder
    light_decoder = LightweightSemanticDecoder(semantic_dim=semantic_dim)
    output_light = light_decoder(image)
    print(f"Lightweight decoder output shape: {output_light.shape}")
    assert output_light.shape == (batch_size, semantic_dim), "Lightweight output shape mismatch"
    
    # Test multi-scale decoder
    multi_decoder = MultiScaleSemanticDecoder(semantic_dim=semantic_dim)
    output_multi = multi_decoder(image)
    print(f"Multi-scale decoder output shape: {output_multi.shape}")
    assert output_multi.shape == (batch_size, semantic_dim), "Multi-scale output shape mismatch"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_semantic_decoder()

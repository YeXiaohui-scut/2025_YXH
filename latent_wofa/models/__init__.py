"""
Latent-WOFA Models Package
"""

from .stage1_codec import PixelNoiseEncoder, PixelNoiseDecoder, Stage1Model, Stage1Loss
from .stage2_embedder import LatentWatermarkEmbedder, NoiseToLatentProjector, AttentionFusion
from .stage2_extractor import PixelWatermarkExtractor
from .distortion_layers import Stage1DistortionLayer, Stage2DistortionLayer, CropAndFuseAttack

__all__ = [
    'PixelNoiseEncoder',
    'PixelNoiseDecoder',
    'Stage1Model',
    'Stage1Loss',
    'LatentWatermarkEmbedder',
    'NoiseToLatentProjector',
    'AttentionFusion',
    'PixelWatermarkExtractor',
    'Stage1DistortionLayer',
    'Stage2DistortionLayer',
    'CropAndFuseAttack',
]
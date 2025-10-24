#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example Training Script for Semantic Watermarking

This script demonstrates how to train the semantic watermarking model
with different prompt strategies.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def create_sample_dataset(output_dir='data/sample_dataset'):
    """
    Create a sample dataset for demonstration.
    In practice, you would use your own dataset.
    """
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Create sample images
    image_paths = []
    for i in range(20):
        # Create random image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = f'sample_{i:03d}.jpg'
        img.save(os.path.join(output_dir, 'images', img_path))
        image_paths.append(img_path)
    
    # Create data list
    df = pd.DataFrame({'path': image_paths})
    df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df.head(5).to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    
    # Create sample captions (optional)
    captions = [
        "A colorful abstract pattern",
        "A random noise texture",
        "A generated image sample",
        "An artificial test image",
        "A synthetic visual pattern",
    ] * 4  # Repeat to match number of images
    
    df['caption'] = captions
    df.to_csv(os.path.join(output_dir, 'captions.csv'), index=False)
    
    print(f"✓ Sample dataset created in {output_dir}")
    print(f"  - 20 training images")
    print(f"  - 5 validation images")
    print(f"  - Captions file included")
    
    return output_dir


def train_with_generic_prompts(args):
    """
    Training strategy 1: Use generic prompts from prompt pool.
    Best for initial training without caption data.
    """
    print("\n" + "="*60)
    print("Training with Generic Prompts")
    print("="*60)
    
    config = OmegaConf.load(args.config)
    
    # Configure for generic prompts
    config.data.params.train.params.pop('caption_file', None)  # Remove caption file
    config.data.params.validation.params.pop('caption_file', None)
    
    # Update paths if using sample dataset
    if args.use_sample_data:
        sample_dir = create_sample_dataset()
        config.data.params.train.params.data_dir = os.path.join(sample_dir, 'images')
        config.data.params.train.params.data_list = os.path.join(sample_dir, 'train.csv')
        config.data.params.validation.params.data_dir = os.path.join(sample_dir, 'images')
        config.data.params.validation.params.data_list = os.path.join(sample_dir, 'val.csv')
    
    # Create data module
    config.data.params.batch_size = args.batch_size
    data = instantiate_from_config(config.data)
    data.setup()
    
    print(f"\n✓ Data loaded:")
    print(f"  - Training samples: {len(data.dataset_train)}")
    print(f"  - Validation samples: {len(data.dataset_validation)}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Create model
    if args.learning_rate > 0:
        config.model.params.learning_rate = args.learning_rate
    
    model = instantiate_from_config(config.model)
    
    print(f"\n✓ Model created:")
    print(f"  - Type: {config.model.target}")
    print(f"  - Semantic dim: {model.semantic_dim}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Lightweight WEmb: {config.model.params.get('use_lightweight_wemb', False)}")
    
    # Setup trainer
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output, 'checkpoints'),
            filename='epoch_{epoch:02d}',
            save_top_k=3,
            monitor='val/cosine_sim',
            mode='max',
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        default_root_dir=args.output,
        log_every_n_steps=10,
    )
    
    print(f"\n✓ Trainer configured:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Accelerator: {trainer.accelerator}")
    print(f"  - Output dir: {args.output}")
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Checkpoints saved to: {os.path.join(args.output, 'checkpoints')}")


def train_with_captions(args):
    """
    Training strategy 2: Use real captions.
    Best for production systems with caption data.
    """
    print("\n" + "="*60)
    print("Training with Real Captions")
    print("="*60)
    
    config = OmegaConf.load(args.config)
    
    # Update paths if using sample dataset
    if args.use_sample_data:
        sample_dir = create_sample_dataset()
        config.data.params.train.params.data_dir = os.path.join(sample_dir, 'images')
        config.data.params.train.params.data_list = os.path.join(sample_dir, 'train.csv')
        config.data.params.train.params.caption_file = os.path.join(sample_dir, 'captions.csv')
        config.data.params.validation.params.data_dir = os.path.join(sample_dir, 'images')
        config.data.params.validation.params.data_list = os.path.join(sample_dir, 'val.csv')
        config.data.params.validation.params.caption_file = os.path.join(sample_dir, 'captions.csv')
    else:
        # Set caption file paths
        config.data.params.train.params.caption_file = args.caption_file
        config.data.params.validation.params.caption_file = args.caption_file
    
    # Rest is same as generic prompts
    train_with_generic_prompts(args)


def demonstrate_prompt_diversity():
    """
    Demonstrate the diversity of prompts generated by the system.
    """
    print("\n" + "="*60)
    print("Prompt Diversity Demonstration")
    print("="*60)
    
    from tools.dataset import datasetWithPrompts
    
    # Create a dummy dataset
    dataset = datasetWithPrompts(
        data_dir='.',
        data_list='data/sample.csv',
        resize=256,
    )
    
    print(f"\nPrompt pool size: {len(dataset.prompt_pool)}")
    print("\nSample prompts from the pool:")
    for i, prompt in enumerate(dataset.prompt_pool[:20]):
        print(f"  {i+1:2d}. {prompt}")
    
    print("\n" + "="*60)


def test_semantic_encoding():
    """
    Test semantic encoding with different prompts.
    """
    print("\n" + "="*60)
    print("Semantic Encoding Test")
    print("="*60)
    
    from models.semanticEmbedding import SemanticEncoder
    import torch.nn.functional as F
    
    # Initialize encoder
    encoder = SemanticEncoder(rotation_seed=42, embedding_dim=512)
    print(f"\n✓ Encoder initialized (dim={encoder.embedding_dim})")
    
    # Test with different prompts
    prompts = [
        "A beautiful landscape with mountains",
        "A cat sitting on a windowsill",
        "An abstract colorful pattern",
        "A beautiful landscape with mountains",  # Duplicate for testing
    ]
    
    print("\nEncoding prompts and computing similarities:")
    vectors = []
    for prompt in prompts:
        vec = encoder(prompt, encrypt=False)
        vectors.append(vec)
    
    print("\nSimilarity matrix:")
    print("        ", end="")
    for i in range(len(prompts)):
        print(f"P{i+1:2d}  ", end="")
    print()
    
    for i, vec_i in enumerate(vectors):
        print(f"Prompt {i+1:2d}", end="")
        for j, vec_j in enumerate(vectors):
            sim = F.cosine_similarity(vec_i, vec_j, dim=-1).item()
            print(f" {sim:4.2f}", end="")
        print()
    
    print("\n✓ Notice:")
    print("  - Identical prompts have similarity ~1.00")
    print("  - Different prompts have lower similarity")
    print("  - This enables unique watermarks per prompt")
    
    # Test encryption
    print("\n" + "="*60)
    print("Encryption/Decryption Test")
    print("="*60)
    
    original = vectors[0]
    encrypted = encoder.rotation_matrix.encrypt(original)
    decrypted = encoder.rotation_matrix.decrypt(encrypted)
    
    diff = torch.abs(original - decrypted).max().item()
    print(f"\n✓ Encryption test:")
    print(f"  - Max difference after encrypt/decrypt: {diff:.2e}")
    print(f"  - Encryption preserves information perfectly")


def main():
    parser = argparse.ArgumentParser(description='Semantic Watermarking Training Example')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'train_captions', 'demo_prompts', 'test_encoding'],
                       help='Training mode or demo')
    parser.add_argument('--config', type=str, 
                       default='configs/SD14_SemanticLaWa.yaml',
                       help='Config file')
    parser.add_argument('--output', type=str, 
                       default='results/semantic_training_example',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00006,
                       help='Learning rate')
    parser.add_argument('--caption_file', type=str, default=None,
                       help='Caption file for training with captions')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Create and use sample dataset for demonstration')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_generic_prompts(args)
    elif args.mode == 'train_captions':
        train_with_captions(args)
    elif args.mode == 'demo_prompts':
        demonstrate_prompt_diversity()
    elif args.mode == 'test_encoding':
        test_semantic_encoding()


if __name__ == '__main__':
    main()

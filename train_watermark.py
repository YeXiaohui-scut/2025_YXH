"""
Training script for Hybrid Watermarking Framework
Stage 1: Train autoencoder on real images (COCO dataset)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import our modules
from models import WatermarkAutoencoder
from utils import (
    load_semantic_extractor, 
    extract_semantic_description,
    generate_watermark_qr,
    save_image
)


# ============================================================================
# Dataset
# ============================================================================

class WatermarkTrainingDataset(Dataset):
    """Dataset for training watermark autoencoder"""
    
    def __init__(
        self, 
        image_dir: str,
        image_size: int = 512,
        semantic_model=None,
        semantic_processor=None,
        device: str = 'cuda'
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.semantic_model = semantic_model
        self.semantic_processor = semantic_processor
        self.device = device
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Extract semantic description using BLIP
        if self.semantic_model is not None:
            text_description = extract_semantic_description(
                image, self.semantic_model, self.semantic_processor, self.device
            )
        else:
            # Fallback: use filename as description
            text_description = img_path.stem
        
        # Generate QR code watermark
        user_id = f"user_{idx % 1000}"
        timestamp = datetime.now().isoformat()
        qr_tensor = generate_watermark_qr(text_description, user_id, timestamp)
        
        return {
            'image': image_tensor,
            'qr_code': qr_tensor,
            'text': text_description,
            'path': str(img_path)
        }


# ============================================================================
# Loss Functions
# ============================================================================

class WatermarkLoss(nn.Module):
    """Combined loss for watermark training"""
    
    def __init__(
        self,
        fidelity_weight: float = 1.0,
        lpips_weight: float = 1.0,
        qr_weight: float = 10.0,
        mask_weight: float = 5.0,
        edge_weight: float = 2.0,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.fidelity_weight = fidelity_weight
        self.lpips_weight = lpips_weight
        self.qr_weight = qr_weight
        self.mask_weight = mask_weight
        self.edge_weight = edge_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # LPIPS loss (perceptual)
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.use_lpips = True
        except:
            print("Warning: LPIPS not available, skipping perceptual loss")
            self.use_lpips = False
    
    def forward(
        self,
        original_image: torch.Tensor,
        watermarked_image: torch.Tensor,
        original_qr: torch.Tensor,
        reconstructed_qr: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor
    ) -> dict:
        """
        Compute all losses
        
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        
        # 1. Fidelity loss (MSE between original and watermarked)
        l_fidelity = self.mse_loss(watermarked_image, original_image)
        losses['fidelity'] = l_fidelity
        
        # 2. LPIPS perceptual loss
        if self.use_lpips:
            # LPIPS expects images in [-1, 1]
            img1 = watermarked_image * 2 - 1
            img2 = original_image * 2 - 1
            l_lpips = self.lpips_fn(img1, img2).mean()
            losses['lpips'] = l_lpips
        else:
            l_lpips = torch.tensor(0.0, device=original_image.device)
            losses['lpips'] = l_lpips
        
        # 3. QR code reconstruction loss (BCE)
        l_qr = self.bce_loss(reconstructed_qr, original_qr)
        losses['qr_reconstruction'] = l_qr
        
        # 4. Tamper mask prediction loss (MSE)
        l_mask_mse = self.mse_loss(pred_mask, gt_mask)
        losses['mask_mse'] = l_mask_mse
        
        # 5. Edge-aware mask loss
        l_mask_edge = self._edge_aware_loss(pred_mask, gt_mask)
        losses['mask_edge'] = l_mask_edge
        
        # Total loss with dynamic weighting
        total_loss = (
            self.fidelity_weight * l_fidelity +
            self.lpips_weight * l_lpips +
            self.qr_weight * l_qr +
            self.mask_weight * l_mask_mse +
            self.edge_weight * l_mask_edge
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def _edge_aware_loss(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """Edge-aware loss for better mask boundary prediction"""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=pred_mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=pred_mask.device).view(1, 1, 3, 3)
        
        # Compute edges
        pred_edge_x = torch.nn.functional.conv2d(pred_mask, sobel_x, padding=1)
        pred_edge_y = torch.nn.functional.conv2d(pred_mask, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2)
        
        gt_edge_x = torch.nn.functional.conv2d(gt_mask, sobel_x, padding=1)
        gt_edge_y = torch.nn.functional.conv2d(gt_mask, sobel_y, padding=1)
        gt_edge = torch.sqrt(gt_edge_x ** 2 + gt_edge_y ** 2)
        
        # Edge loss
        edge_loss = self.mse_loss(pred_edge, gt_edge)
        
        return edge_loss
    
    def update_weights(self, epoch: int, max_epochs: int):
        """
        Dynamic loss weight adjustment (GenPTW strategy)
        Early training: focus on extraction (qr and mask)
        Late training: focus on fidelity
        """
        progress = epoch / max_epochs
        
        # Decrease extraction weights, increase fidelity weight
        self.qr_weight = 10.0 * (1 - 0.5 * progress)
        self.mask_weight = 5.0 * (1 - 0.5 * progress)
        self.fidelity_weight = 1.0 + 2.0 * progress
        
        print(f"Epoch {epoch}: Updated loss weights - "
              f"fidelity={self.fidelity_weight:.2f}, "
              f"qr={self.qr_weight:.2f}, "
              f"mask={self.mask_weight:.2f}")


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: WatermarkLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'fidelity': 0.0,
        'lpips': 0.0,
        'qr_reconstruction': 0.0,
        'mask_mse': 0.0,
        'mask_edge': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        image = batch['image'].to(device)
        qr_code = batch['qr_code'].to(device)
        
        # Forward pass
        watermarked, recon_qr, pred_mask, gt_mask = model(image, qr_code)
        
        # Compute losses
        losses = criterion(image, watermarked, qr_code, recon_qr, gt_mask, pred_mask)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'qr': f"{losses['qr_reconstruction'].item():.4f}",
            'fid': f"{losses['fidelity'].item():.4f}"
        })
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    return total_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: WatermarkLoss,
    device: str
) -> dict:
    """Validate model"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'fidelity': 0.0,
        'lpips': 0.0,
        'qr_reconstruction': 0.0,
        'mask_mse': 0.0,
        'mask_edge': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            image = batch['image'].to(device)
            qr_code = batch['qr_code'].to(device)
            
            watermarked, recon_qr, pred_mask, gt_mask = model(image, qr_code)
            
            losses = criterion(image, watermarked, qr_code, recon_qr, gt_mask, pred_mask)
            
            for key in total_losses:
                total_losses[key] += losses[key].item()
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= len(dataloader)
    
    return total_losses


def main(args):
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BLIP model for semantic extraction
    print("\n" + "="*60)
    print("Loading BLIP model...")
    print("="*60)
    try:
        semantic_model, semantic_processor = load_semantic_extractor(device)
    except:
        print("Warning: Could not load BLIP, using fallback mode")
        semantic_model = None
        semantic_processor = None
    
    # Create dataset
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    
    train_dataset = WatermarkTrainingDataset(
        image_dir=args.data_dir,
        image_size=args.image_size,
        semantic_model=semantic_model,
        semantic_processor=semantic_processor,
        device=device
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = WatermarkAutoencoder(hidden_dim=args.hidden_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    
    # Create loss function
    criterion = WatermarkLoss(device=device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Update loss weights
        criterion.update_weights(epoch, args.epochs)
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Print losses
        print(f"\nTrain Losses:")
        for key, value in train_losses.items():
            print(f"  {key}: {value:.6f}")
        
        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
            }, best_path)
            print(f"Best model saved: {best_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Watermarking Framework")
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='outputs/watermark_training',
                       help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for models')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)

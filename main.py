"""
Main inference script for Hybrid Watermarking Framework
Stage 2: Apply trained watermark to T2I generated images and verify
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image

# Import our modules
from models import WatermarkEmbedder, WatermarkExtractor
from utils import (
    load_semantic_extractor,
    load_t2i_model,
    extract_semantic_description,
    generate_watermark_qr,
    verify_watermark_qr,
    compute_similarity,
    compute_bert_similarity,
    tensor_to_pil,
    pil_to_tensor,
    save_image
)


class WatermarkPipeline:
    """Complete watermark generation and verification pipeline"""
    
    def __init__(
        self,
        embedder_checkpoint: str,
        extractor_checkpoint: str,
        device: str = 'cuda'
    ):
        self.device = device
        
        # Load models
        print("Loading watermark models...")
        self.embedder = WatermarkEmbedder(hidden_dim=64).to(device)
        self.extractor = WatermarkExtractor(hidden_dim=64).to(device)
        
        # Load checkpoints
        self._load_checkpoint(embedder_checkpoint, self.embedder)
        self._load_checkpoint(extractor_checkpoint, self.extractor)
        
        self.embedder.eval()
        self.extractor.eval()
        
        # Load BLIP for semantic extraction
        print("Loading BLIP model...")
        try:
            self.semantic_model, self.semantic_processor = load_semantic_extractor(device)
        except Exception as e:
            print(f"Warning: Could not load BLIP: {e}")
            self.semantic_model = None
            self.semantic_processor = None
        
        # Load T2I model (lazy loading)
        self.t2i_pipeline = None
        
        print("✓ Pipeline ready!")
    
    def _load_checkpoint(self, checkpoint_path: str, model):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized weights (for testing only)")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    def load_t2i_model(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Lazy load T2I model"""
        if self.t2i_pipeline is None:
            print("Loading T2I model (this may take a while)...")
            self.t2i_pipeline = load_t2i_model(model_id, self.device)
    
    def generate_and_watermark(
        self,
        prompt: str,
        user_id: str,
        private_key_path: str = "keys/private_key.pem",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image with T2I and embed watermark
        
        Args:
            prompt: Text prompt for image generation
            user_id: User identifier
            private_key_path: Path to private key
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            output_path: Optional path to save watermarked image
            
        Returns:
            tuple: (watermarked_image_tensor, metadata)
        """
        print("\n" + "="*60)
        print("Generate and Watermark")
        print("="*60)
        
        # Ensure T2I model is loaded
        self.load_t2i_model()
        
        # Generate image
        print(f"\n1. Generating image with prompt: '{prompt}'")
        with torch.no_grad():
            generated = self.t2i_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        print(f"   ✓ Image generated: {generated.size}")
        
        # Extract semantic description
        print("\n2. Extracting semantic description...")
        if self.semantic_model is not None:
            semantic_text = extract_semantic_description(
                generated, 
                self.semantic_model, 
                self.semantic_processor,
                self.device
            )
        else:
            semantic_text = prompt  # Fallback to prompt
        
        print(f"   Semantic: {semantic_text}")
        
        # Generate watermark QR code
        print("\n3. Generating watermark QR code...")
        timestamp = datetime.now().isoformat()
        qr_tensor = generate_watermark_qr(
            semantic_text, 
            user_id, 
            timestamp, 
            private_key_path
        )
        qr_tensor = qr_tensor.to(self.device)
        print(f"   ✓ QR code generated: {qr_tensor.shape}")
        
        # Convert generated image to tensor
        image_tensor = pil_to_tensor(generated, self.device)
        
        # Resize if needed
        if image_tensor.shape[2] != 512 or image_tensor.shape[3] != 512:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(512, 512), mode='bilinear', align_corners=False
            )
        
        # Embed watermark
        print("\n4. Embedding watermark...")
        with torch.no_grad():
            watermarked_tensor = self.embedder(image_tensor, qr_tensor)
        
        print(f"   ✓ Watermark embedded: {watermarked_tensor.shape}")
        
        # Save if output path provided
        if output_path:
            save_image(watermarked_tensor, output_path)
        
        # Prepare metadata
        metadata = {
            'prompt': prompt,
            'semantic_text': semantic_text,
            'user_id': user_id,
            'timestamp': timestamp,
            'output_path': output_path
        }
        
        print("\n" + "="*60)
        print("✓ Generation and watermarking complete!")
        print("="*60)
        
        return watermarked_tensor, metadata
    
    def verify_image(
        self,
        image_path: str,
        public_key_path: str = "keys/public_key.pem",
        use_bert_similarity: bool = False
    ) -> dict:
        """
        Verify watermarked image
        
        Args:
            image_path: Path to watermarked image
            public_key_path: Path to public key
            use_bert_similarity: Use BERT for semantic similarity
            
        Returns:
            dict: Verification results
        """
        print("\n" + "="*60)
        print("Image Verification")
        print("="*60)
        
        # Load image
        print(f"\n1. Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = pil_to_tensor(image, self.device)
        
        # Resize if needed
        if image_tensor.shape[2] != 512 or image_tensor.shape[3] != 512:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(512, 512), mode='bilinear', align_corners=False
            )
        
        print(f"   ✓ Image loaded: {image_tensor.shape}")
        
        # Extract watermark and tamper mask
        print("\n2. Extracting watermark and tamper mask...")
        with torch.no_grad():
            extracted_qr, tamper_mask = self.extractor(image_tensor)
        
        print(f"   ✓ Extracted QR: {extracted_qr.shape}")
        print(f"   ✓ Tamper mask: {tamper_mask.shape}")
        
        # Check for tampering
        print("\n3. Checking for tampering...")
        max_tamper_score = tamper_mask.max().item()
        mean_tamper_score = tamper_mask.mean().item()
        tamper_detected = max_tamper_score > 0.5
        
        tamper_result = {
            'detected': tamper_detected,
            'max_score': max_tamper_score,
            'mean_score': mean_tamper_score,
            'tampered_pixels': (tamper_mask > 0.5).sum().item()
        }
        
        if tamper_detected:
            print(f"   ⚠ WARNING: Possible tampering detected!")
            print(f"   Max score: {max_tamper_score:.4f}")
            print(f"   Mean score: {mean_tamper_score:.4f}")
        else:
            print(f"   ✓ No significant tampering detected")
            print(f"   Max score: {max_tamper_score:.4f}")
        
        # Verify cryptographic signature
        print("\n4. Verifying cryptographic signature...")
        is_valid, decoded_message = verify_watermark_qr(extracted_qr, public_key_path)
        
        signature_result = {
            'valid': is_valid,
            'message': decoded_message
        }
        
        if is_valid:
            print(f"   ✓ Signature is VALID!")
            print(f"   Original text: {decoded_message['text_data']}")
            print(f"   User ID: {decoded_message['user_id']}")
            print(f"   Timestamp: {decoded_message['timestamp']}")
        else:
            print(f"   ✗ Signature is INVALID or QR code corrupted!")
        
        # Verify semantic consistency
        print("\n5. Verifying semantic consistency...")
        semantic_result = {}
        
        if is_valid and self.semantic_model is not None:
            # Extract current semantic description
            current_semantic = extract_semantic_description(
                image,
                self.semantic_model,
                self.semantic_processor,
                self.device
            )
            
            original_semantic = decoded_message['text_data']
            
            # Compute similarity
            if use_bert_similarity:
                similarity = compute_bert_similarity(
                    current_semantic, 
                    original_semantic,
                    self.device
                )
            else:
                similarity = compute_similarity(current_semantic, original_semantic)
            
            semantic_result = {
                'current': current_semantic,
                'original': original_semantic,
                'similarity': similarity,
                'consistent': similarity > 0.5
            }
            
            print(f"   Current semantic: {current_semantic}")
            print(f"   Original semantic: {original_semantic}")
            print(f"   Similarity: {similarity:.4f}")
            
            if semantic_result['consistent']:
                print(f"   ✓ Semantic content is consistent")
            else:
                print(f"   ⚠ WARNING: Semantic content may have changed")
        else:
            print(f"   ⚠ Skipping semantic verification (BLIP not available or signature invalid)")
            semantic_result = {'available': False}
        
        # Generate final report
        print("\n" + "="*60)
        print("Verification Report")
        print("="*60)
        
        results = {
            'image_path': image_path,
            'tampering': tamper_result,
            'signature': signature_result,
            'semantic': semantic_result
        }
        
        # Overall verdict
        all_checks_passed = (
            not tamper_result['detected'] and
            signature_result['valid'] and
            (semantic_result.get('consistent', True) if semantic_result.get('available', False) else True)
        )
        
        results['verdict'] = 'AUTHENTIC' if all_checks_passed else 'SUSPICIOUS'
        
        print(f"\nFinal Verdict: {results['verdict']}")
        print("="*60 + "\n")
        
        return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Watermarking Framework - Inference and Verification"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate and watermark image')
    gen_parser.add_argument('--prompt', type=str, required=True,
                           help='Text prompt for image generation')
    gen_parser.add_argument('--user_id', type=str, default='default_user',
                           help='User identifier')
    gen_parser.add_argument('--output', type=str, required=True,
                           help='Output path for watermarked image')
    gen_parser.add_argument('--private_key', type=str, default='keys/private_key.pem',
                           help='Path to private key')
    gen_parser.add_argument('--embedder_checkpoint', type=str, required=True,
                           help='Path to embedder checkpoint')
    gen_parser.add_argument('--extractor_checkpoint', type=str, required=True,
                           help='Path to extractor checkpoint')
    gen_parser.add_argument('--steps', type=int, default=50,
                           help='Number of inference steps')
    gen_parser.add_argument('--guidance', type=float, default=7.5,
                           help='Guidance scale')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify watermarked image')
    verify_parser.add_argument('--image', type=str, required=True,
                              help='Path to image to verify')
    verify_parser.add_argument('--public_key', type=str, default='keys/public_key.pem',
                              help='Path to public key')
    verify_parser.add_argument('--extractor_checkpoint', type=str, required=True,
                              help='Path to extractor checkpoint')
    verify_parser.add_argument('--use_bert', action='store_true',
                              help='Use BERT for semantic similarity')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # Initialize pipeline
        pipeline = WatermarkPipeline(
            embedder_checkpoint=args.embedder_checkpoint,
            extractor_checkpoint=args.extractor_checkpoint
        )
        
        # Generate and watermark
        watermarked, metadata = pipeline.generate_and_watermark(
            prompt=args.prompt,
            user_id=args.user_id,
            private_key_path=args.private_key,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            output_path=args.output
        )
        
        print(f"\n✓ Success! Watermarked image saved to: {args.output}")
        
    elif args.command == 'verify':
        # Initialize pipeline (only need extractor)
        pipeline = WatermarkPipeline(
            embedder_checkpoint='',  # Not needed for verification
            extractor_checkpoint=args.extractor_checkpoint
        )
        
        # Verify image
        results = pipeline.verify_image(
            image_path=args.image,
            public_key_path=args.public_key,
            use_bert_similarity=args.use_bert
        )
        
        print(f"\n✓ Verification complete!")
        print(f"Verdict: {results['verdict']}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    print("="*70)
    print(" Hybrid Watermarking Framework - Inference & Verification")
    print(" Combines MetaSeal + SWIFT + GenPTW")
    print("="*70 + "\n")
    
    main()

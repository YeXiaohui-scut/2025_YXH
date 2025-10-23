"""
Semantic Watermarking Inference Script
Generates images with semantic watermarks embedded from text prompts
Based on the upgraded SemanticLaWa system
"""
import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from pytorch_lightning import seed_everything

from models.semanticEmbedding import SemanticEncoder
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips
from utils_img import no_ssl_verification


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_from_config(config, ckpt, verbose=False):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def generate_watermarked_image(
    model_sd, 
    model_lawa,
    semantic_encoder,
    prompt, 
    sampler,
    seed,
    ddim_steps=50,
    scale=7.5,
    H=512,
    W=512,
    metadata=None,
):
    """
    Generate image with semantic watermark
    Args:
        model_sd: Stable Diffusion model
        model_lawa: Semantic LaWa watermarking model
        semantic_encoder: Semantic encoder for text
        prompt: Text prompt for image generation
        sampler: DDIM/PLMS sampler
        seed: Random seed
        ddim_steps: Number of diffusion steps
        scale: Classifier-free guidance scale
        H, W: Image dimensions
        metadata: Optional metadata to embed
    Returns:
        original_image, watermarked_image, semantic_vector
    """
    seed_everything(seed)
    
    # Encode prompt to conditioning
    with torch.no_grad():
        uc = None
        if scale != 1.0:
            uc = model_sd.get_learned_conditioning([""])
        c = model_sd.get_learned_conditioning([prompt])
        
        shape = [4, H // 8, W // 8]
        
        # Generate latent with diffusion
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0.0
        )
        
        # Decode without watermark (for comparison)
        x_samples_original = model_sd.decode_first_stage(samples_ddim)
        x_samples_original = torch.clamp((x_samples_original + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Generate semantic vector from prompt
        semantic_vector = semantic_encoder(prompt, encrypt=True, metadata=metadata)
        
        # Apply watermark using SemanticLaWa
        # Prepare latent for watermarking
        x_watermark = samples_ddim / model_lawa.scale_factor
        x_watermark = model_lawa.ae.post_quant_conv(x_watermark)
        
        # Generate watermarked image
        x_samples_watermarked = model_lawa(x_watermark, x_samples_original, semantic_vector)
        x_samples_watermarked = torch.clamp((x_samples_watermarked + 1.0) / 2.0, min=0.0, max=1.0)
        
    return x_samples_original, x_samples_watermarked, semantic_vector


def extract_and_verify_watermark(
    watermarked_image,
    original_prompt,
    semantic_encoder,
    decoder,
    metadata=None,
    threshold=0.85,
):
    """
    Extract and verify semantic watermark from image
    Args:
        watermarked_image: Watermarked image tensor (0-1 range)
        original_prompt: Original prompt used for watermarking
        semantic_encoder: Semantic encoder
        decoder: Watermark decoder
        metadata: Optional metadata
        threshold: Similarity threshold for verification
    Returns:
        is_authentic, similarity_score
    """
    with torch.no_grad():
        # Normalize image for decoder (expects ImageNet normalization)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        watermarked_normalized = normalize(watermarked_image)
        
        # Extract semantic vector
        extracted_semantic = decoder(watermarked_normalized)
        
        # Decrypt
        decrypted_semantic = semantic_encoder.rotation_matrix.decrypt(extracted_semantic)
        
        # Verify
        is_authentic, similarity = semantic_encoder.verify(
            decrypted_semantic,
            original_prompt,
            threshold=threshold,
            metadata=metadata
        )
        
    return is_authentic, similarity


def main(args):
    """Main inference function"""
    seed_everything(args.seed)
    
    # Load Stable Diffusion model
    print("Loading Stable Diffusion model...")
    config_sd = OmegaConf.load(args.config_sd)
    with no_ssl_verification():
        model_sd = load_model_from_config(config_sd, args.ckpt_sd)
    model_sd = model_sd.to(device)
    
    # Setup sampler
    if args.sampler == "ddim":
        sampler = DDIMSampler(model_sd)
    elif args.sampler == "plms":
        sampler = PLMSSampler(model_sd)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")
    
    # Load Semantic LaWa model
    print("Loading Semantic LaWa model...")
    config_lawa = OmegaConf.load(args.config_lawa)
    model_lawa = instantiate_from_config(config_lawa.model)
    
    if args.ckpt_lawa:
        print(f"Loading LaWa checkpoint from {args.ckpt_lawa}")
        checkpoint = torch.load(args.ckpt_lawa, map_location="cpu")
        model_lawa.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model_lawa = model_lawa.to(device)
    model_lawa.eval()
    
    # Get semantic encoder and decoder
    semantic_encoder = model_lawa.semantic_encoder
    decoder = model_lawa.decoder
    
    # Prepare metadata if specified
    metadata = None
    if args.add_metadata:
        metadata = {
            'model_version': args.model_version,
            'timestamp': args.timestamp,
        }
        if args.user_id:
            metadata['user_id'] = args.user_id
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Generate image
    print(f"Generating image for prompt: '{args.prompt}'")
    original_img, watermarked_img, semantic_vec = generate_watermarked_image(
        model_sd=model_sd,
        model_lawa=model_lawa,
        semantic_encoder=semantic_encoder,
        prompt=args.prompt,
        sampler=sampler,
        seed=args.seed,
        ddim_steps=args.ddim_steps,
        scale=args.scale,
        H=args.H,
        W=args.W,
        metadata=metadata,
    )
    
    # Save images
    original_img_pil = Image.fromarray(
        (original_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    watermarked_img_pil = Image.fromarray(
        (watermarked_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    
    base_name = args.prompt.replace(" ", "_")[:50]
    original_path = os.path.join(args.outdir, f"{base_name}_original.png")
    watermarked_path = os.path.join(args.outdir, f"{base_name}_watermarked.png")
    
    original_img_pil.save(original_path)
    watermarked_img_pil.save(watermarked_path)
    
    print(f"Saved original image to: {original_path}")
    print(f"Saved watermarked image to: {watermarked_path}")
    
    # Compute quality metrics
    print("\n=== Image Quality Metrics ===")
    psnr = compute_psnr(original_img, watermarked_img)
    ssim = compute_ssim(original_img, watermarked_img)
    mse = compute_mse(original_img, watermarked_img)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"MSE: {mse:.6f}")
    
    # Extract and verify watermark
    print("\n=== Watermark Verification ===")
    is_authentic, similarity = extract_and_verify_watermark(
        watermarked_img,
        args.prompt,
        semantic_encoder,
        decoder,
        metadata=metadata,
        threshold=args.threshold,
    )
    
    print(f"Verification Result: {'✓ AUTHENTIC' if is_authentic else '✗ NOT AUTHENTIC'}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: {args.threshold}")
    
    # Save semantic vector if requested
    if args.save_semantic_vector:
        semantic_path = os.path.join(args.outdir, f"{base_name}_semantic.pt")
        torch.save(semantic_vec, semantic_path)
        print(f"Saved semantic vector to: {semantic_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Watermarking Inference")
    
    # Model paths
    parser.add_argument("--config_sd", type=str, default="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
                        help="Path to Stable Diffusion config")
    parser.add_argument("--ckpt_sd", type=str, default="weights/stable-diffusion-v1/model.ckpt",
                        help="Path to Stable Diffusion checkpoint")
    parser.add_argument("--config_lawa", type=str, default="configs/SD14_SemanticLaWa_inference.yaml",
                        help="Path to Semantic LaWa config")
    parser.add_argument("--ckpt_lawa", type=str, default="",
                        help="Path to Semantic LaWa checkpoint")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("--H", type=int, default=512,
                        help="Image height")
    parser.add_argument("--W", type=int, default=512,
                        help="Image width")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "plms"],
                        help="Sampling method")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Metadata parameters
    parser.add_argument("--add_metadata", action="store_true",
                        help="Add metadata to watermark")
    parser.add_argument("--model_version", type=str, default="SemanticLaWa-v1.0",
                        help="Model version to embed")
    parser.add_argument("--user_id", type=str, default="",
                        help="User ID to embed")
    parser.add_argument("--timestamp", type=str, default="",
                        help="Timestamp to embed")
    
    # Verification parameters
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for verification")
    
    # Output parameters
    parser.add_argument("--outdir", type=str, default="results/semantic_watermarking",
                        help="Output directory")
    parser.add_argument("--save_semantic_vector", action="store_true",
                        help="Save semantic vector to file")
    
    args = parser.parse_args()
    
    # Auto-set timestamp if not provided
    if args.add_metadata and not args.timestamp:
        from datetime import datetime
        args.timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    main(args)

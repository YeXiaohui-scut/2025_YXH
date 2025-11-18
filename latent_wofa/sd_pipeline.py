"""
Stable Diffusion 集成管道
核心功能：在SD生成过程中嵌入水印，并从生成图像中提取水印
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from typing import Optional, List, Union


class WatermarkedStableDiffusionPipeline:
    """
    带水印的Stable Diffusion生成管道
    在VAE潜空间注入水印，实现生成+嵌入一体化
    """
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        vae_model_id="stabilityai/sd-vae-ft-mse",
        device="cuda",
        dtype=torch.float16
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading Stable Diffusion pipeline from {model_id}...")
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            vae_model_id
        ).to(device, dtype=dtype)
        
        # 加载UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet"
        ).to(device, dtype=dtype)
        
        # 加载文本编码器
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder"
        ).to(device, dtype=dtype)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )
        
        # 加载调度器
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        
        # 设置为评估模式
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        # 水印嵌入器和提取器（需要外部加载）
        self.watermark_embedder = None
        self.watermark_extractor = None
        self.stage1_encoder = None
        self.stage1_decoder = None
        
        print("✅ Stable Diffusion pipeline loaded successfully!")
    
    def load_watermark_models(self, embedder, extractor, stage1_encoder, stage1_decoder):
        """
        加载水印模型
        Args:
            embedder: LatentWatermarkEmbedder
            extractor: PixelWatermarkExtractor
            stage1_encoder: PixelNoiseEncoder
            stage1_decoder: PixelNoiseDecoder
        """
        self.watermark_embedder = embedder.to(self.device, dtype=self.dtype)
        self.watermark_extractor = extractor.to(self.device)
        self.stage1_encoder = stage1_encoder.to(self.device)
        self.stage1_decoder = stage1_decoder.to(self.device)
        
        self.watermark_embedder.eval()
        self.watermark_extractor.eval()
        self.stage1_encoder.eval()
        self.stage1_decoder.eval()
        
        print("✅ Watermark models loaded successfully!")
    
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        编码文本提示词
        """
        # 编码正向提示词
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # 编码负向提示词
        if negative_prompt:
            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
        else:
            uncond_inputs = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
        
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]
        
        # 拼接 [negative, positive]
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        return text_embeddings
    
    @torch.no_grad()
    def generate_with_watermark(
        self,
        prompt: Union[str, List[str]],
        watermark_bits: torch.Tensor,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        return_latents: bool = False
    ):
        """
        生成带水印的图像（核心方法）
        """
        if self.watermark_embedder is None:
            raise ValueError("Please load watermark models first using load_watermark_models()")
        
        batch_size = watermark_bits.size(0)
        
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # 1. 编码文本提示词
        text_embeddings = self.encode_prompt(prompt, negative_prompt)
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        
        # 2. 准备初始噪声
        latent_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # 3. 编码水印到像素噪声
        with torch.no_grad():
            w_noise = self.stage1_encoder(watermark_bits)
        
        # 4. 嵌入水印到初始潜码（关键步骤！）
        latents = self.watermark_embedder(
            latents.to(torch.float32), 
            w_noise.to(torch.float32)
        ).to(self.dtype)
        
        # 5. 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 6. 去噪循环（标准DDIM采样）
        for i, t in enumerate(self.scheduler.timesteps):
            # 扩展潜码用于CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # UNet预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 调度器步进
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 7. VAE解码到像素空间
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        
        # 8. 后处理
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        if return_latents:
            return pil_images, latents
        return pil_images
    
    @torch.no_grad()
    def extract_watermark(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]):
        """
        从图像中提取水印
        """
        if self.watermark_extractor is None:
            raise ValueError("Please load watermark models first")
        
        # 转换为tensor
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            images_np = [np.array(img.resize((512, 512))) for img in images]
            images_tensor = torch.from_numpy(np.stack(images_np)).permute(0, 3, 1, 2).float()
            images_tensor = images_tensor / 127.5 - 1.0
        else:
            images_tensor = images
        
        images_tensor = images_tensor.to(self.device)
        
        # 1. 提取像素噪声
        w_noise_pred = self.watermark_extractor(images_tensor)
        
        # 2. 解码比特串
        watermark_bits = self.stage1_decoder(w_noise_pred)
        
        return watermark_bits
    
    def encode_image_to_latent(self, image: Union[Image.Image, torch.Tensor]):
        """
        将图像编码到VAE潜空间
        用于Stage II训练
        """
        if isinstance(image, Image.Image):
            image = np.array(image.resize((512, 512)))
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            image = image / 127.5 - 1.0
        
        image = image.to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * 0.18215
        
        return latent
    
    def decode_latent_to_image(self, latent: torch.Tensor):
        """
        将VAE潜码解码为图像
        用于Stage II训练
        """
        latent = 1 / 0.18215 * latent
        
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

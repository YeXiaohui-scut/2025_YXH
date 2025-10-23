"""
Semantic LaWa: Upgraded LaWa with semantic watermarking
Replaces binary watermarks with semantic vectors
Based on LatentSeal, SWIFT, and MetaSeal approaches
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from lpips.lpips import LPIPS

from models.semanticEmbedding import SemanticEncoder, RotationMatrix
from models.unetWEmb import MultiScaleUNetWEmb, LightweightUNetWEmb
from models.semanticDecoder import SemanticDecoder


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class SemanticLaWa(pl.LightningModule):
    """
    Semantic LaWa: Uses semantic vectors instead of binary watermarks
    """
    def __init__(self,
                 first_stage_config,
                 decoder_config,
                 discriminator_config,
                 semantic_encoder_config=None,
                 recon_type='rgb',
                 learning_rate=0.0001,
                 epoch_num=100,
                 recon_loss_weight=2.0,
                 adversarial_loss_weight=2.0,
                 perceptual_loss_weight=0.0,
                 lpips_loss_weights_path=None,
                 semantic_loss_weight=2.0,
                 ramp=100000,
                 watermark_addition_weight=0.1,
                 noise_config='__none__',
                 use_ema=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 extraction_resize=False,
                 start_attack_acc_thresh=0.85,
                 dis_update_freq=1,
                 rotation_seed=42,
                 use_lightweight_wemb=False,
                 cosine_similarity_threshold=0.85,
                 ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.scale_factor = scale_factor
        self.extraction_resize = extraction_resize
        
        # Load autoencoder
        self.ae = instantiate_from_config(first_stage_config)
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False
        
        # Semantic encoder
        if semantic_encoder_config is not None:
            self.semantic_encoder = instantiate_from_config(semantic_encoder_config)
        else:
            self.semantic_encoder = SemanticEncoder(rotation_seed=rotation_seed)
        
        self.semantic_encoder.eval()
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False
        
        self.semantic_dim = self.semantic_encoder.embedding_dim
        
        # Semantic decoder (watermark extractor)
        self.decoder = instantiate_from_config(decoder_config)
        
        # Discriminator
        self.discriminator = instantiate_from_config(discriminator_config)
        
        # WEmb modules - using U-Net for perturbation generation
        if use_lightweight_wemb:
            # Lightweight version for faster training
            self.wemb_initial_0 = LightweightUNetWEmb(
                semantic_dim=self.semantic_dim,
                feature_channels=4,
                base_channels=16
            )
            self.wemb_initial = LightweightUNetWEmb(
                semantic_dim=self.semantic_dim,
                feature_channels=512,
                base_channels=32
            )
            self.wemb_layers = nn.ModuleList([
                LightweightUNetWEmb(semantic_dim=self.semantic_dim, feature_channels=128, base_channels=32),
                LightweightUNetWEmb(semantic_dim=self.semantic_dim, feature_channels=256, base_channels=32),
                LightweightUNetWEmb(semantic_dim=self.semantic_dim, feature_channels=512, base_channels=32),
                LightweightUNetWEmb(semantic_dim=self.semantic_dim, feature_channels=512, base_channels=32),
            ])
        else:
            # Full U-Net version
            from models.unetWEmb import UNetWEmb
            self.wemb_initial_0 = UNetWEmb(
                semantic_dim=self.semantic_dim,
                feature_channels=4,
                base_channels=16
            )
            self.wemb_initial = UNetWEmb(
                semantic_dim=self.semantic_dim,
                feature_channels=512,
                base_channels=32
            )
            self.wemb_layers = nn.ModuleList([
                UNetWEmb(semantic_dim=self.semantic_dim, feature_channels=128, base_channels=32),
                UNetWEmb(semantic_dim=self.semantic_dim, feature_channels=256, base_channels=32),
                UNetWEmb(semantic_dim=self.semantic_dim, feature_channels=512, base_channels=32),
                UNetWEmb(semantic_dim=self.semantic_dim, feature_channels=512, base_channels=32),
            ])
        
        self.watermark_addition_weight = watermark_addition_weight
        self.cosine_similarity_threshold = cosine_similarity_threshold
        
        # Training state
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_semantic = None
        self.register_buffer("fixed_input", torch.tensor(False))
        self.register_buffer("noise_activated", torch.tensor(False))
        self.noise_active_step = 0
        self.start_attack_acc_thresh = start_attack_acc_thresh
        self.dis_update_freq = dis_update_freq
        self.use_ema = use_ema
        
        # Noise/augmentation
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        
        # Load checkpoint if provided
        if ckpt_path != '__none__':
            print("###############################################################################")
            print("Using provided model weights!")
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        
        # Normalization
        self.normalize_vqgan_to_imagenet = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Loss parameters
        self.recon_type = recon_type
        if self.recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1, 100, 100]).unsqueeze(1).float())
        
        self.recon_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        
        if lpips_loss_weights_path is not None:
            self.perceptual_loss = LPIPS(weights_path=lpips_loss_weights_path)
            self.perceptual_loss.eval()
        elif self.perceptual_loss_weight > 0:
            self.perceptual_loss = LPIPS()
            self.perceptual_loss.eval()
        
        self.adversarial_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def forward(self, x, image, semantic_vector):
        """
        Forward pass with semantic watermarking
        Args:
            x: Latent representation (batch_size, 4, H, W)
            image: Original image for reference
            semantic_vector: Encrypted semantic vector (batch_size, semantic_dim)
        Returns:
            watermarked_image
        """
        # Initial latent perturbation
        dx = self.wemb_initial_0(semantic_vector, x)
        x_new = x + dx * self.watermark_addition_weight
        
        # Set decoder shape
        self.ae.decoder.last_z_shape = x_new.shape
        
        # VAE decoder with watermark injection
        temb = None
        
        # Initial convolution
        h = self.ae.decoder.conv_in(x_new)
        
        # Middle blocks
        h = self.ae.decoder.mid.block_1(h, temb)
        h = self.ae.decoder.mid.attn_1(h)
        h = self.ae.decoder.mid.block_2(h, temb)
        
        # Add semantic watermark to middle features
        dh = self.wemb_initial(semantic_vector, h)
        h = h + dh * self.watermark_addition_weight
        
        # Upsampling layers with watermark injection
        layer_outputs = []
        
        # Layer 3
        i_level = 3
        for i_block in range(self.ae.decoder.num_res_blocks + 1):
            h = self.ae.decoder.up[i_level].block[i_block](h, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h = self.ae.decoder.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.ae.decoder.up[i_level].upsample(h)
        dh = self.wemb_layers[0](semantic_vector, h)
        h = h + dh * self.watermark_addition_weight
        
        # Layer 2
        i_level = 2
        for i_block in range(self.ae.decoder.num_res_blocks + 1):
            h = self.ae.decoder.up[i_level].block[i_block](h, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h = self.ae.decoder.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.ae.decoder.up[i_level].upsample(h)
        dh = self.wemb_layers[1](semantic_vector, h)
        h = h + dh * self.watermark_addition_weight
        
        # Layer 1
        i_level = 1
        for i_block in range(self.ae.decoder.num_res_blocks + 1):
            h = self.ae.decoder.up[i_level].block[i_block](h, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h = self.ae.decoder.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.ae.decoder.up[i_level].upsample(h)
        dh = self.wemb_layers[2](semantic_vector, h)
        h = h + dh * self.watermark_addition_weight
        
        # Layer 0
        i_level = 0
        for i_block in range(self.ae.decoder.num_res_blocks + 1):
            h = self.ae.decoder.up[i_level].block[i_block](h, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h = self.ae.decoder.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.ae.decoder.up[i_level].upsample(h)
        dh = self.wemb_layers[3](semantic_vector, h)
        h = h + dh * self.watermark_addition_weight
        
        # Final output
        if self.ae.decoder.give_pre_end:
            return h
        
        h = self.ae.decoder.norm_out(h)
        h = self.nonlinearity(h)
        h = self.ae.decoder.conv_out(h)
        if self.ae.decoder.tanh_out:
            h = torch.tanh(h)
        
        return h
    
    @torch.no_grad()
    def get_input(self, batch, bs=None):
        """Get input data from batch"""
        image = batch['image']
        prompt = batch.get('prompt', None)  # For semantic watermarking
        
        if bs is not None:
            image = image[:bs]
            if prompt is not None:
                prompt = prompt[:bs]
        else:
            bs = image.shape[0]
        
        # Prepare image
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        
        # Encode image to latent
        x = self.encode_first_stage(image).detach()
        x = self.ae.post_quant_conv(1. / self.scale_factor * x).detach()
        
        # Get reconstruction without watermark
        image_rec = self.ae.decoder(x).detach()
        image_rec = torch.clamp(image_rec, min=-1., max=1.)
        
        # Generate semantic vector
        if prompt is not None:
            # Use provided prompts
            semantic_vector = self.semantic_encoder(prompt, encrypt=True)
        else:
            # Generate random semantic vectors for training
            # (In practice, you would use real prompts)
            semantic_vector = torch.randn(bs, self.semantic_dim, device=image.device)
            # Apply rotation to simulate encryption
            semantic_vector = self.semantic_encoder.rotation_matrix.encrypt(semantic_vector)
        
        semantic_vector = semantic_vector.detach()
        
        out = [x, semantic_vector, image, image_rec, prompt]
        return out
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, semantic_vec, img, img_rec_gt, prompts = self.get_input(batch)
        
        real_labels = torch.ones(img_rec_gt.size(0), 1).type_as(x)
        fake_labels = torch.zeros(img_rec_gt.size(0), 1).type_as(x)
        
        # Train watermark embedder and extractor
        if optimizer_idx == 0:
            loss_dict = {}
            
            # Create watermarked image
            image_rec = self(x, img_rec_gt, semantic_vec)
            
            # Reconstruction loss
            rec_loss = self.compute_recon_loss(img_rec_gt.contiguous(), image_rec.contiguous())
            
            # PSNR
            psnr = self.calculate_psnr(image_rec, img_rec_gt)
            
            # Perceptual loss
            if self.perceptual_loss_weight > 0:
                p_loss = self.perceptual_loss(img_rec_gt.contiguous(), image_rec.contiguous()).mean()
                loss_dict['emb_p_loss'] = p_loss
            else:
                p_loss = 0
            
            # Discriminator loss
            d_on_watermarked_image = self.discriminator(image_rec)
            loss_adversarial = (-1) * d_on_watermarked_image.mean()
            
            # Apply noise/augmentation
            if hasattr(self, 'noise') and self.noise_activated:
                image_rec_noisy = self.noise(image_rec, self.global_step, active_step=self.noise_active_step, p=0.9)
            else:
                image_rec_noisy = image_rec
            
            # Extract semantic watermark
            pred_semantic = self.decoder(image_rec_noisy)
            
            # Semantic loss - cosine similarity with target
            # Decrypt the predicted semantic vector
            pred_semantic_decrypted = self.semantic_encoder.rotation_matrix.decrypt(pred_semantic)
            
            # Get target semantic (decrypt the input)
            target_semantic = self.semantic_encoder.rotation_matrix.decrypt(semantic_vec)
            
            # Cosine similarity loss (maximize similarity)
            cosine_sim = F.cosine_similarity(pred_semantic_decrypted, target_semantic, dim=-1)
            semantic_loss = 1 - cosine_sim.mean()  # Convert to loss (minimize distance)
            
            # Total loss
            w_emb_loss = (rec_loss * self.recon_weight + 
                         self.adversarial_loss_weight * loss_adversarial +
                         semantic_loss * self.semantic_loss_weight +
                         self.perceptual_loss_weight * p_loss)
            
            # Compute accuracy (similarity above threshold)
            accuracy = (cosine_sim > self.cosine_similarity_threshold).float().mean()
            
            # Update loss dict
            loss_dict['cosine_sim'] = cosine_sim.mean()
            loss_dict['accuracy'] = accuracy
            loss_dict['psnr'] = psnr
            loss_dict['emb_loss'] = w_emb_loss
            loss_dict['emb_rec_loss'] = rec_loss
            loss_dict['emb_semantic_loss'] = semantic_loss
            loss_dict['emb_adversarial_loss'] = loss_adversarial
            
            # Activate attacks when accuracy is high enough
            if (accuracy > self.start_attack_acc_thresh) and (not self.noise_activated):
                print(f'Activating attack at step {self.global_step}')
                self.noise_activated = ~self.noise_activated
                self.noise_active_step = self.global_step
            
            # Logging
            loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
            return w_emb_loss
        
        # Train discriminator
        if optimizer_idx == 1:
            loss_dict = {}
            
            # Generate watermarked images
            image_rec = self(x, img_rec_gt, semantic_vec)
            
            # Discriminator predictions
            fake_preds = self.discriminator(image_rec).mean()
            real_preds = self.discriminator(img_rec_gt).mean()
            
            # Clip weights (Wasserstein GAN)
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            
            # Discriminator loss
            d_loss = fake_preds - real_preds
            
            loss_dict['dis_loss'] = d_loss
            loss_dict['dis_real_loss'] = real_preds
            loss_dict['dis_fake_loss'] = fake_preds
            
            loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            return d_loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_dict = {}
            x, semantic_vec, img, img_rec_gt, prompts = self.get_input(batch)
            
            # Generate watermarked image
            image_rec = self(x, img_rec_gt, semantic_vec)
            
            # Reconstruction loss
            rec_loss = self.compute_recon_loss(img_rec_gt.contiguous(), image_rec.contiguous())
            psnr = self.calculate_psnr(image_rec, img_rec_gt)
            
            # Perceptual loss
            if self.perceptual_loss_weight > 0:
                p_loss = self.perceptual_loss(img_rec_gt.contiguous(), image_rec.contiguous()).mean()
                loss_dict['emb_p_loss'] = p_loss
            else:
                p_loss = 0
            
            # Discriminator
            d_on_watermarked_image = self.discriminator(image_rec)
            loss_adversarial = (-1.) * d_on_watermarked_image.mean()
            real_preds = self.discriminator(img_rec_gt).mean()
            fake_preds = self.discriminator(image_rec).mean()
            
            # Apply noise
            if hasattr(self, 'noise') and self.noise_activated:
                image_rec_noisy = self.noise(image_rec, self.global_step, active_step=self.noise_active_step, p=0.99)
            else:
                image_rec_noisy = image_rec
            
            # Extract and verify watermark
            pred_semantic = self.decoder(image_rec_noisy)
            pred_semantic_decrypted = self.semantic_encoder.rotation_matrix.decrypt(pred_semantic)
            target_semantic = self.semantic_encoder.rotation_matrix.decrypt(semantic_vec)
            
            cosine_sim = F.cosine_similarity(pred_semantic_decrypted, target_semantic, dim=-1)
            semantic_loss = 1 - cosine_sim.mean()
            accuracy = (cosine_sim > self.cosine_similarity_threshold).float().mean()
            
            # Total losses
            w_emb_loss = (rec_loss * self.recon_weight +
                         self.adversarial_loss_weight * loss_adversarial +
                         semantic_loss * self.semantic_loss_weight +
                         self.perceptual_loss_weight * p_loss)
            
            d_loss = fake_preds - real_preds
            
            # Update loss dict
            loss_dict['cosine_sim'] = cosine_sim.mean()
            loss_dict['accuracy'] = accuracy
            loss_dict['psnr'] = psnr
            loss_dict['emb_loss'] = w_emb_loss
            loss_dict['emb_rec_loss'] = rec_loss
            loss_dict['emb_semantic_loss'] = semantic_loss
            loss_dict['emb_adversarial_loss'] = loss_adversarial
            loss_dict['dis_loss'] = d_loss
            loss_dict['dis_real_loss'] = real_preds
            loss_dict['dis_fake_loss'] = fake_preds
            
            loss_dict_val = {f"val/{key}": val for key, val in loss_dict.items()}
            self.log_dict(loss_dict_val, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        # Collect all embedding parameters
        embedding_params = (list(self.wemb_initial_0.parameters()) +
                           list(self.wemb_initial.parameters()) +
                           list(self.wemb_layers.parameters()) +
                           list(self.decoder.parameters()))
        
        discriminator_params = list(self.discriminator.parameters())
        
        embedding_optimizer = torch.optim.AdamW(embedding_params, lr=self.learning_rate)
        discriminator_optimizer = torch.optim.AdamW(discriminator_params, lr=self.learning_rate)
        
        embedding_lr_scheduler = CosineAnnealingLR(embedding_optimizer, T_max=self.epoch_num)
        discriminator_lr_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max=self.epoch_num)
        
        if self.dis_update_freq == 0:
            return [embedding_optimizer, discriminator_optimizer]
        elif self.dis_update_freq > 0:
            return [
                {
                    "optimizer": embedding_optimizer,
                    "frequency": 1,
                },
                {
                    "optimizer": discriminator_optimizer,
                    "frequency": self.dis_update_freq,
                },
            ]
    
    # Helper functions
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            rec_loss = torch.mean((inputs - reconstructions) ** 2)
        elif self.recon_type == 'yuv':
            reconstructions_yuv = self.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = self.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv) ** 2, dim=[2, 3])
            rec_loss = torch.mean(torch.mm(yuv_loss, self.yuv_scales))
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss
    
    def calculate_psnr(self, image_rec, img_rec_gt):
        with torch.no_grad():
            delta = 255 * torch.clamp((image_rec + 1.0) / 2.0 - (img_rec_gt + 1.0) / 2.0, 0, 1)
            delta = delta.reshape(-1, image_rec.shape[-3], image_rec.shape[-2], image_rec.shape[-1])
            psnr = 20 * np.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))
            psnr = psnr.mean()
        return psnr
    
    def rgb_to_yuv(self, image):
        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        
        return torch.stack([y, u, v], -3)
    
    def nonlinearity(self, x):
        return x * torch.sigmoid(x)

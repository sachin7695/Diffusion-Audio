# This file was auto-generated from v1.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import os
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import itertools
import torch

class Sampler:

    def __init__(self, num_training_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_training_steps = num_training_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = self.linear_beta_schedule()
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)

    def linear_beta_schedule(self):
        """
        Creates the β schedule as a linear ramp from beta_start → beta_end.
        
        Returns: Tensor of shape (T,) 
        e.g. [0.0001, 0.0001018, ..., 0.02]
        """
        return torch.linspace(self.beta_start, self.beta_end, self.num_training_steps)

    def _repeated_unsqueeze(self, target_shape, input):
        """
        Expands `coeff` to have the same number of dimensions as `target_tensor`
        by repeatedly appending trailing size-1 dimensions.
        WHY THIS IS NEEDED (Broadcasting Problem):
        ------------------------------------------
        `coeff` after indexing α_bar has shape: (batch_size,)       ← 1D
        `target_tensor` (the image) has shape:  (batch_size, C, H, W) ← 4D
        PyTorch aligns shapes FROM THE RIGHT when broadcasting:
          coeff        : (batch_size,)
          image        : (batch_size, C, H, W)
          ↳ Would align as: (1, 1, 1, batch_size) — multiplied against WIDTH ❌
        After this function:
          coeff        : (batch_size, 1, 1, 1)
          image        : (batch_size, C, H, W)
          ↳ Aligns as  : scalar per image × every pixel of that image ✅
        Args:
            target_tensor : The tensor whose rank (ndim) we want to match
            coeff         : The 1D coefficient tensor to expand
        Returns:
            coeff reshaped to (batch_size, 1, 1, ...) 
        """
        while target_shape.dim() > input.dim():
            input = input.unsqueeze(-1)
        return input

    def add_noise(self, inputs, timesteps):
        """
        FORWARD DIFFUSION — given a clean image x₀, produce a noisy version x_t.
        Math:  x_t = √ᾱ(t) · x₀  +  √(1-ᾱ(t)) · ε       ε ~ N(0, I)
        Used DURING TRAINING to generate (noisy_image, actual_noise) pairs.
        The network learns to predict the actual_noise from the noisy_image.
        Args:
            inputs    : Clean image tensor, shape (batch, C, H, W)
            timesteps : 1D tensor of timestep indices, shape (batch,)
        Returns:
            noisy_image : Corrupted image at step t
            noise       : The actual noise that was added (training target)
        """
        batch_size, c, h, w = inputs.shape
        device = inputs.device
        alpha_cumulative_prod_timesteps = self.alpha_cumulative_prod[timesteps].to(device)
        ' \n        tensor([37,  4, 58, 81])\n        tensor([0.9823,         0.9993,          0.9608,           0.9283])\n                B1.             B2                 B3                B4\n                till-T37.       till-T4.         till-T58\n                cum prod ᾱ(t).  cum prod ᾱ(t).     cum prod ᾱ(t)\n                0.9823          0.9993             0.9608\n\n        '
        mean_coeff = alpha_cumulative_prod_timesteps ** 0.5
        var_coeff = (1 - alpha_cumulative_prod_timesteps) ** 0.5
        '\n        shape of mean torch.Size([4])\n        shape of input torch.Size([4, 3, 64, 64])\n\n        so we need to reshape the mean as per the input shape \n        one solution is \n        mean = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # shape [4, 1, 1, 1] == input.shape\n\n        robust solution create a function and by a loop match the dimension\n        thats what _repeated_unsqueeze() does \n        '
        mean_coeff = self._repeated_unsqueeze(inputs, mean_coeff)
        var_coeff = self._repeated_unsqueeze(inputs, var_coeff)
        noise = torch.randn_like(inputs)
        mean = mean_coeff * inputs
        var = var_coeff * noise
        noisy_image = mean + var
        return (noisy_image, noise)

    def remove_noise(self, input, timestep, predicted_noise):
        """
        REVERSE DIFFUSION — given a noisy image x_t and the model's predicted noise,
        take ONE denoising step backward to get x_{t-1}.
        Math (DDPM reverse step):
            μ  = (1/√α_t) · (x_t − (β_t / √(1-ᾱ_t)) · ε_θ)
            σ² = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
            x_{t-1} = μ + σ · z       z ~ N(0, I)   (z = 0 at t=0)
        Called DURING GENERATION, going from t=999 down to t=0.

        
        Args:
            input           : Noisy image x_t,     shape (batch, C, H, W)
            timestep        : Current step index t, shape (batch,)
            predicted_noise : ε_θ predicted by the neural net, shape (batch, C, H, W)
        Returns:
            denoised: Slightly cleaner image x_{t-1}
        """
        assert input.shape == predicted_noise.shape, 'Shapes of noise pattern and input image must be identical!!'
        b, c, h, w = input.shape
        device = input.device
        equal_to_zero_mask = timestep == 0
        beta_t = self.beta_schedule[timestep].to(device)
        alpha_t = self.alpha[timestep].to(device)
        alpha_cumulative_prod_t = self.alpha_cumulative_prod[timestep].to(device)
        alpha_cumulative_prod_t_prev = self.alpha_cumulative_prod[timestep - 1].to(device)
        alpha_cumulative_prod_t_prev[equal_to_zero_mask] = 1
        noise = torch.randn_like(input)
        variance = beta_t * (1 - alpha_cumulative_prod_t_prev) / (1 - alpha_cumulative_prod_t)
        variance = self._repeated_unsqueeze(input, variance)
        sigma_t_z = noise * variance ** 0.5
        noise_coefficient = beta_t / (1 - alpha_cumulative_prod_t) ** 0.5
        noise_coefficient = self._repeated_unsqueeze(input, noise_coefficient)
        reciprocal_root_a_t = alpha_t ** (-0.5)
        reciprocal_root_a_t = self._repeated_unsqueeze(input, reciprocal_root_a_t)
        mean = denoised = reciprocal_root_a_t * (input - noise_coefficient * predicted_noise)
        denoised = mean + sigma_t_z
        return denoised

class SelfAttention(nn.Module):

    def __init__(self, in_channel, num_heads=12, att_p=0, proj_p=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q = nn.Linear(in_channel, in_channel)
        self.k = nn.Linear(in_channel, in_channel)
        self.v = nn.Linear(in_channel, in_channel)
        self.att_p = att_p
        self.proj = nn.Linear(in_channel, in_channel)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        b, seq_len, d_in = x.shape
        query = self.q(x).reshape(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.k(x).reshape(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.v(x).reshape(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        context_vec = F.scaled_dot_product_attention(query, key, value, dropout_p=self.att_p)
        context_vec = context_vec.permute(0, 2, 1, 3).reshape(b, seq_len, d_in)
        proj = self.proj(context_vec)
        proj = self.proj_drop(proj)
        return proj

class MLP(nn.Module):

    def __init__(self, in_channel, mlp_ratio=4, mlp_p=0):
        super().__init__()
        self.in_channel = in_channel
        self.mlp_ratio = mlp_ratio
        self.mlp_p = mlp_p
        self.fc1 = nn.Linear(in_channel, in_channel * mlp_ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(in_channel * mlp_ratio, in_channel)
        self.drop2 = nn.Dropout(mlp_p)

    def forward(self, x):
        b, seq_len, in_channel = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, in_channel, num_heads=4, mlp_ratio=4, mlp_p=0, att_p=0, proj_p=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channel, eps=1e-06)
        self.attn = SelfAttention(in_channel, num_heads, att_p, proj_p)
        self.norm2 = nn.LayerNorm(in_channel, eps=1e-06)
        self.mlp = MLP(in_channel, mlp_ratio, mlp_p)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        x = x.permute(0, 2, 1)
        b, seq_len, in_channel = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, h, w)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groupnorm_num_groups, time_embed_dim):
        super().__init__()
        self.time_expand = nn.Linear(time_embed_dim, out_channels)
        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.resize_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same') if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embeddings):
        residual_connection = x
        time_embed = self.time_expand(time_embeddings)
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        time_embeddings = time_embed.unsqueeze(-1).unsqueeze(-1)
        x = x + time_embeddings
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.resize_channels(residual_connection)
        return x

class UNET(nn.Module):

    def __init__(self, start_dim=64, in_channels=3, out_channels=64, dim_mults=(1, 2, 4), residual_blocks_per_group=1, groupnorm_num_groups=16, time_embed_dim=128):
        super().__init__()
        self.input_image_channels = in_channels
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = (channel_sizes[0], channel_sizes[-1])
        self.encoder_config = []
        for idx, d in enumerate(channel_sizes):
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), 'residual'))
            self.encoder_config.append(((d, d), 'downsample'))
            self.encoder_config.append((d, 'attention'))
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d, channel_sizes[idx + 1]), 'residual'))
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), 'residual'))
        out_dim = ending_channel_size
        reversed_encoder_config = self.encoder_config[::-1]
        self.decoder_config = []
        for idx, (metadata, l_type) in enumerate(reversed_encoder_config):
            if l_type != 'attention':
                enc_in_channels, enc_out_channels = metadata
                self.decoder_config.append(((out_dim + enc_out_channels, enc_in_channels), 'residual'))
                if l_type == 'downsample':
                    self.decoder_config.append(((enc_in_channels, enc_in_channels), 'upsample'))
                out_dim = enc_in_channels
            else:
                in_channels = metadata
                self.decoder_config.append((in_channels, 'attention'))
        self.decoder_config.append(((starting_channel_size * 2, starting_channel_size), 'residual'))
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, starting_channel_size, kernel_size=3, padding='same')
        self.encoder = nn.ModuleList()
        for metadata, l_type in self.encoder_config:
            if l_type == 'residual':
                in_ch, out_ch = metadata
                self.encoder.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, groupnorm_num_groups=groupnorm_num_groups, time_embed_dim=time_embed_dim))
            elif l_type == 'downsample':
                in_ch, out_ch = metadata
                self.encoder.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1))
            elif l_type == 'attention':
                in_ch = metadata
                self.encoder.append(TransformerBlock(in_channel=in_ch))
        self.bottleneck = nn.ModuleList()
        for (in_channels, out_channels), _ in self.bottleneck_config:
            self.bottleneck.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, groupnorm_num_groups=groupnorm_num_groups, time_embed_dim=time_embed_dim))
        self.decoder = nn.ModuleList()
        for metadata, l_type in self.decoder_config:
            if l_type == 'residual':
                in_ch, out_ch = metadata
                self.decoder.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, groupnorm_num_groups=groupnorm_num_groups, time_embed_dim=time_embed_dim))
            elif l_type == 'upsample':
                in_channels, out_channels = metadata
                self.decoder.append(UpSampleBlock(in_channels=in_channels, out_channels=out_channels))
            elif l_type == 'attention':
                in_channels = metadata
                self.decoder.append(TransformerBlock(in_channels))
        self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size, out_channels=self.input_image_channels, kernel_size=3, padding='same')

    def forward(self, x, time_embeddings):
        residuals = []
        x = self.conv_in_proj(x)
        residuals.append(x)
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, time_embeddings)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):
                x = module(x)
                residuals.append(x)
            else:
                x = module(x)
        for module in self.bottleneck:
            x = module(x, time_embeddings)
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                residual_tensor = residuals.pop()
                x = torch.cat([x, residual_tensor], axis=1)
                x = module(x, time_embeddings)
            else:
                x = module(x)
        x = self.conv_out_proj(x)
        return x

class Diffusion(nn.Module):

    def __init__(self, in_channels=3, start_dim=64, dim_mults=(1, 2, 4, 4), residual_blocks_per_group=1, groupnorm_num_groups=16, time_embed_dim=128, time_embed_dim_ratio=2):
        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.residual_blocks_per_group = residual_blocks_per_group
        self.groupnorm_num_groups = groupnorm_num_groups
        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim * time_embed_dim_ratio)
        self.sinusoid_time_embeddings = SinusodialTimeEmbeddings(time_embed_dim=self.time_embed_dim, scaled_time_embed_dim=self.scaled_time_embed_dim)
        self.unet = UNET(in_channels=in_channels, start_dim=start_dim, dim_mults=dim_mults, residual_blocks_per_group=residual_blocks_per_group, groupnorm_num_groups=groupnorm_num_groups, time_embed_dim=self.scaled_time_embed_dim)

    def forward(self, noisy_inputs, timesteps):
        timestep_embeddings = self.sinusoid_time_embeddings(timesteps)
        noise_pred = self.unet(noisy_inputs, timestep_embeddings)
        return noise_pred

class SinusodialTimeEmbeddings(nn.Module):
    def __init__(self, time_embed_dim, scaled_time_embed_dim):

        super().__init__() 
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, time_embed_dim, 2).float() / time_embed_dim)), requires_grad=False)
        # if time_embed_dim = 128,  64 indexes for sin and 64 for cosine  
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, scaled_time_embed_dim),
            nn.SiLU(),
            nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim),
            nn.SiLU()
            )  

    def forward(self, timesteps):
        # timesteps.shape = (b,)
        timestep_freqs = timesteps.unsqueeze(1) * self.inv_freq.unsqueeze(0) # pos * inv_freq
        embeddings = torch.cat([torch.sin(timestep_freqs), torch.cos(timestep_freqs)], dim=-1) # shape [B, time_embed_dim]
        embeddings = self.time_mlp(embeddings) # shape (b, scaled_time_embed_dim)

        return embeddings

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        )

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        upsampled = self.upsample(inputs)
        assert (upsampled.shape == (batch, channels, height*2, width*2))
        return upsampled
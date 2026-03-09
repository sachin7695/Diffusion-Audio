# Diffusion Models for Audio & Image Generation

This repository contains implementations of Diffusion Models programmed from scratch in PyTorch. The models are designed to generate both images (faces) and audio using purely convolutional/attention-based architectures (UNET) and Denoising Diffusion Probabilistic Models (DDPM).

## 🗂️ Project Structure

- `src/`
  - `notebooks/v1.ipynb`: The original notebook walking through the math and implementation of DDPM for image generation (CelebA dataset).
  - `diffusion_models.py`: Modularized PyTorch code containing the core network (`UNET`), the scheduling components (`Sampler`), `SelfAttention`, `TransformerBlocks`, and custom positional `SinusoidalTimeEmbeddings`.
  - `audio_diffusion_train.py`: An end-to-end training and inference script that adapts the 2D UNET to generate Audio by representing sound as Mel-Spectrograms and vocoding via Griffin-Lim.
  - `compute_global_stats.py`: A script to compute global statistics of the dataset.

- `data/`
  - `download_emilia_en.py`: A script to stream and download the Emilia-EN audio dataset from HuggingFace to use as raw training data.
  - *Note: Raw datasets, audio checkpoints, and generated media are ignored in source control due to size constraints.*

## 🧠 Core Architecture Highlights

* **Denoising Framework**: Pure implementation of the reverse diffusion process, gradually removing Gaussian noise to synthesize structure.
* **UNET Backbone**: Features Downsampling and Upsampling convolutions coupled with Bottleneck layers.
* **Conditioning**: Time step variables $(t)$ are encoded using sinusoidal embeddings, passed through MLPs, and injected into the Residual Blocks via scale/shift (SiLU conditioning).
* **Self-Attention**: Multi-head self-attention mechanisms built into the Transformer blocks to allow the network to learn global context and long-range dependencies, crucial for high-quality image and audio generation.

## 🎧 Audio Generation via Spectrograms

Generating audio is accomplished by treating audio signals as images.
1. **Transform**: 1D `.wav` files are converted into 2D Mel-Spectrograms (Log-scale Decibels).
2. **Standardization**: Spectrograms are squeezed into fixed sizes (e.g., $128 \times 128$) representing a fixed duration (e.g., 5 seconds with a specific hop length).
3. **Diffusion**: The UNET treats the normalized spectrogram as a 1-channel grayscale image and learns to denoise it.
4. **Vocoding**: The generated 2D Mel-Spectrogram is converted back to linear amplitude and passed through an Inverse Mel Scale and the Griffin-Lim algorithm to construct the final 1D `.wav` file.

## 🚀 Getting Started

**Audio Training Example:**
```bash
python src/audio_diffusion_clean.py
```
This will automatically construct a PyTorch Dataloader, initialize the UNET and AdamW optimizer, and begin training the diffusion network on the provided metadata CSV. Checkpoints and generated samples will be saved periodically according to the `--eval_interval` logic.

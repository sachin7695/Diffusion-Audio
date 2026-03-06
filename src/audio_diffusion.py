import os
import warnings
import torch
import torchaudio
import pandas as pd
import torch.nn as nn
import torchaudio.transforms as T

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

from diffusion_models import Diffusion, Sampler


class AudioConfig:
    SAMPLE_RATE = 24000
    N_MELS = 128
    MAX_FRAMES = 128

    AUDIO_SECONDS = 5
    TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)

    HOP_LENGTH = 944
    N_FFT = 2048

    BATCH_SIZE = 8
    LR = 5e-4
    TRAIN_STEPS = 5000
    DIFFUSION_STEPS = 1000
    EVAL_INTERVAL = 100

    GENERATED_DIR = "/Users/sachin/Documents/diffusion/data/audio_data/generated_audio"
    CKPT_DIR = "/Users/sachin/Documents/diffusion/data/audio_data/checkpoints"


mel_transform = T.MelSpectrogram(
    sample_rate=AudioConfig.SAMPLE_RATE,
    n_fft=AudioConfig.N_FFT,
    hop_length=AudioConfig.HOP_LENGTH,
    n_mels=AudioConfig.N_MELS,
    power=2.0,
)
inverse_mel = T.InverseMelScale(
    n_stft=AudioConfig.N_FFT // 2 + 1,
    n_mels=AudioConfig.N_MELS,
    sample_rate=AudioConfig.SAMPLE_RATE,
)

griffin = T.GriffinLim(
    n_fft=AudioConfig.N_FFT,
    hop_length=AudioConfig.HOP_LENGTH,
)
db_transform = T.AmplitudeToDB()


def audio_to_spec(waveform):
    mel = mel_transform(waveform)
    mel = db_transform(mel)

    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    mel = mel * 2 - 1
    return mel


def spec_to_audio(mel_norm, orig_min=-100, orig_max=20):
    mel = (mel_norm + 1) / 2
    mel = mel * (orig_max - orig_min) + orig_min

    mel = 10 ** (mel / 20)

    linear = inverse_mel(mel)
    waveform = griffin(linear)

    return waveform


class AudioDataset(Dataset):

    def __init__(self, metadata_csv):
        df = pd.read_csv(metadata_csv, sep="|")
        df = df[(df.duration >= 4.0) & (df.duration <= 10.0)]

        self.paths = df.filepath.tolist()
        print(f"Loaded {len(self.paths)} audio files")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        if wav.shape[1] > AudioConfig.TARGET_SAMPLES:
            wav = wav[:, :AudioConfig.TARGET_SAMPLES]
        else:
            pad = AudioConfig.TARGET_SAMPLES - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        spec = audio_to_spec(wav)
        return spec


def generate_sample(model, sampler, device, step):

    model.eval()
    print(f"\nGenerating sample at step {step}")

    x = torch.randn(
        1, 1,
        AudioConfig.N_MELS,
        AudioConfig.MAX_FRAMES
    ).to(device)

    with torch.no_grad():

        for t in reversed(range(AudioConfig.DIFFUSION_STEPS)):
            ts = torch.full((1,), t, device=device)

            noise_pred = model(x, ts)

            x = sampler.remove_noise(
                x.cpu(),
                ts.cpu(),
                noise_pred.cpu()
            )

            x = x.to(device)

    os.makedirs(AudioConfig.GENERATED_DIR, exist_ok=True)

    spec = x.cpu().squeeze(0)
    audio = spec_to_audio(spec)

    out = os.path.join(AudioConfig.GENERATED_DIR, f"audio_{step}.wav")
    torchaudio.save(out, audio, AudioConfig.SAMPLE_RATE)

    print("Saved:", out)
    model.train()


def train():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    dataset = AudioDataset(
        "/Users/sachin/Documents/diffusion/data/audio_data/metadata.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=AudioConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    model = Diffusion(in_channels=1, start_dim=64).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=AudioConfig.LR)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2500,
        num_training_steps=AudioConfig.TRAIN_STEPS,
    )

    sampler = Sampler(AudioConfig.DIFFUSION_STEPS)
    loss_fn = nn.MSELoss()

    os.makedirs(AudioConfig.CKPT_DIR, exist_ok=True)

    progress = tqdm(range(AudioConfig.TRAIN_STEPS))

    step = 0
    model.train()

    while step < AudioConfig.TRAIN_STEPS:

        for specs in loader:

            b = specs.shape[0]

            t = torch.randint(0, AudioConfig.DIFFUSION_STEPS, (b,))

            noisy, noise = sampler.add_noise(specs, t)

            pred = model(noisy.to(device), t.to(device))
            loss = loss_fn(pred, noise.to(device))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress.update(1)
            progress.set_postfix(loss=loss.item())

            step += 1

            if step % AudioConfig.EVAL_INTERVAL == 0:

                generate_sample(model, sampler, device, step)

                ckpt = os.path.join(
                    AudioConfig.CKPT_DIR,
                    f"diffusion_{step}.pth"
                )

                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "loss": loss.item()
                }, ckpt)

                print("Checkpoint saved:", ckpt)

            if step >= AudioConfig.TRAIN_STEPS:
                break


if __name__ == "__main__":
    train()
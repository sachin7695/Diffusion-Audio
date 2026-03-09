import os
import warnings
import torch
import torchaudio
import pandas as pd
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")

from diffusion_models import Diffusion, Sampler


class AudioConfig:
    SAMPLE_RATE = 24000
    N_MELS      = 128       # matches 20k checkpoint
    MAX_FRAMES  = 128       # matches 20k checkpoint

    AUDIO_SECONDS  = 5
    TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)   # 120,000

    HOP_LENGTH = 944        # matches 20k checkpoint — DO NOT change
    N_FFT      = 2048

    BATCH_SIZE = 8
    LR         = 5e-5       # low LR for fine-tuning from a converged checkpoint

    RESUME_CKPT = "/data/TTS/sachin-data/exp/Diffusion-Audio/data/audio_data/checkpoints/diffusion_20000.pth"
    TRAIN_STEPS = 60000     # 20k already done → 40k new steps

    DIFFUSION_STEPS = 1000
    EVAL_INTERVAL   = 1000
    EVAL_BATCHES    = 20
    GEN_STEPS       = 1000

    GL_ITERS = 128          # more iterations = cleaner phase reconstruction

    DB_MIN = -48.8   # 1st percentile across 12,612 files
    DB_MAX = 50.6    # 99th percentile across 12,612 files


    GENERATED_DIR = "/data/TTS/sachin-data/exp/Diffusion-Audio/data/audio_data/generated_audio"
    CKPT_DIR      = "/data/TTS/sachin-data/exp/Diffusion-Audio/data/audio_data/checkpoints"
    PLOTS_DIR     = "/data/TTS/sachin-data/exp/Diffusion-Audio/data/audio_data/plots"


# ──────────────────────────────────────────────────────────
# Transforms — MUST match exactly what 20k ckpt was trained on
# NO norm/mel_scale changes — those would break the resumed model
# ──────────────────────────────────────────────────────────
mel_transform = T.MelSpectrogram(
    sample_rate=AudioConfig.SAMPLE_RATE,
    n_fft=AudioConfig.N_FFT,
    hop_length=AudioConfig.HOP_LENGTH,
    n_mels=AudioConfig.N_MELS,
    power=2.0,
    # norm and mel_scale intentionally NOT set — matches original training
)
griffin = T.GriffinLim(
    n_fft=AudioConfig.N_FFT,
    hop_length=AudioConfig.HOP_LENGTH,
    n_iter=AudioConfig.GL_ITERS,
    momentum=0.99,
)
db_transform = T.AmplitudeToDB(top_db=80)

# Pseudoinverse — safe for any N_MELS, replaces InverseMelScale
_mel_fb_pinv = torch.linalg.pinv(mel_transform.mel_scale.fb.T).float()


# def audio_to_spec(waveform):
#     mel = mel_transform(waveform)
#     mel = db_transform(mel)
#     mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
#     mel = mel * 2 - 1
#     return mel

def audio_to_spec(waveform):
    mel = mel_transform(waveform)
    mel = db_transform(mel)
    # FIXED normalization (replaces per-sample min/max)
    mel = (mel - AudioConfig.DB_MIN) / (AudioConfig.DB_MAX - AudioConfig.DB_MIN)
    mel = mel.clamp(0, 1)  # clip outliers
    mel = mel * 2 - 1      # scale to [-1, 1]
    return mel



def spec_to_audio(mel_norm):
    """
    Invert normalised mel spectrogram back to waveform.
    orig_min/max must match top_db=80 → range is -80 to 0 dB.
    """
    mel = (mel_norm + 1) / 2
    mel = mel * (AudioConfig.DB_MAX - AudioConfig.DB_MIN) + AudioConfig.DB_MIN
    mel = 10.0 ** (mel / 20.0)

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)

    fb     = _mel_fb_pinv.to(mel.device).unsqueeze(0)
    linear = torch.relu(torch.bmm(fb, mel))
    return griffin(linear)


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, metadata_csv):
        df = pd.read_csv(metadata_csv, sep="|")
        # Wide range — more data prevents overfitting
        # Short clips padded, long clips randomly cropped
        df = df[(df.duration >= 4.0) & (df.duration <= 15.0)]
        self.paths = df.filepath.tolist()
        print(f"Loaded {len(self.paths)} audio files (4–15s)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.paths[idx])

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        if sr != AudioConfig.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, AudioConfig.SAMPLE_RATE)

        if wav.shape[1] > AudioConfig.TARGET_SAMPLES:
            max_start = wav.shape[1] - AudioConfig.TARGET_SAMPLES
            start = torch.randint(0, max_start + 1, (1,)).item()
            wav = wav[:, start: start + AudioConfig.TARGET_SAMPLES]
        else:
            wav = torch.nn.functional.pad(wav, (0, AudioConfig.TARGET_SAMPLES - wav.shape[1]))

        return audio_to_spec(wav)


# ──────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────
def plot_losses(train_steps, train_losses, eval_steps, eval_losses,
                lr_steps, lr_values, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def smooth(values, alpha=0.97):
        smoothed, last = [], None
        for v in values:
            last = v if last is None else alpha * last + (1 - alpha) * v
            smoothed.append(last)
        return smoothed

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_steps, smooth(train_losses), color="#4C72B0",
            linewidth=1.5, label="Train loss (EMA smoothed)")
    ax.plot(train_steps, train_losses, color="#4C72B0", linewidth=0.3, alpha=0.2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, color="#DD8452",
                linewidth=2, marker="o", markersize=5, label="Eval loss")
    ax.axvline(x=20000, color="gray", linestyle="--", alpha=0.5, label="Resume (20k)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Diffusion Model — Training & Eval Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lr_steps, lr_values, color="#55A868", linewidth=1.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("LR Schedule (Cosine with Warmup)", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Plots saved -> {save_dir}")


# ──────────────────────────────────────────────────────────
# Eval loss
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def compute_eval_loss(model, loader, sampler, loss_fn, device, n_batches=20):
    model.eval()
    total, count = 0.0, 0
    for i, specs in enumerate(loader):
        if i >= n_batches:
            break
        b = specs.shape[0]
        t = torch.randint(0, AudioConfig.DIFFUSION_STEPS, (b,))
        noisy, noise = sampler.add_noise(specs, t)
        total += loss_fn(model(noisy.to(device), t.to(device)), noise.to(device)).item()
        count += 1
    model.train()
    return total / max(count, 1)


# ──────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────
def generate_sample(model, sampler, device, step):
    model.eval()
    print(f"\n  Generating 5s audio at step {step} ...")

    x = torch.randn(1, 1, AudioConfig.N_MELS, AudioConfig.MAX_FRAMES).to(device)

    timesteps = list(reversed(range(
        0, AudioConfig.DIFFUSION_STEPS,
        AudioConfig.DIFFUSION_STEPS // AudioConfig.GEN_STEPS
    )))

    with torch.no_grad():
        for t_val in timesteps:
            ts = torch.full((1,), t_val, device=device)
            noise_pred = model(x, ts)
            x = sampler.remove_noise(x.cpu(), ts.cpu(), noise_pred.cpu()).to(device)

    os.makedirs(AudioConfig.GENERATED_DIR, exist_ok=True)
    spec = x.cpu().squeeze(0)
    audio = spec_to_audio(spec)
    out_path = os.path.join(AudioConfig.GENERATED_DIR, f"audio_{step}.wav")
    torchaudio.save(out_path, audio, AudioConfig.SAMPLE_RATE)
    print(f"  Saved -> {out_path}  ({audio.shape[-1]/AudioConfig.SAMPLE_RATE:.1f}s)")
    model.train()


# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────
def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")      # Linux/Windows GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       # Mac GPU
    else:
        device = torch.device("cpu")       # fallback

    print(f"Using device: {device}")

    dataset = AudioDataset(
        "/data/TTS/sachin-data/exp/Diffusion-Audio/data/audio_data/metadata.csv"
    )
    loader = DataLoader(
        dataset, batch_size=AudioConfig.BATCH_SIZE,
        shuffle=True, num_workers=16, pin_memory=True,
    )

    model     = Diffusion(in_channels=1, start_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=AudioConfig.LR, weight_decay=1e-4)
    sampler   = Sampler(AudioConfig.DIFFUSION_STEPS)
    loss_fn   = nn.MSELoss()

    train_steps, train_losses = [], []
    eval_steps,  eval_losses  = [], []
    lr_steps,    lr_values    = [], []

    # ── Resume ───────────────────────────────────────────
    start_step = 0
    if AudioConfig.RESUME_CKPT and os.path.exists(AudioConfig.RESUME_CKPT):
        print(f"\nResuming from: {AudioConfig.RESUME_CKPT}")
        ckpt       = torch.load(AudioConfig.RESUME_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        # Intentionally NOT loading optimizer state — fresh low LR
        start_step   = ckpt["step"]
        train_steps  = ckpt.get("train_steps",  [])
        train_losses = ckpt.get("train_losses", [])
        eval_steps   = ckpt.get("eval_steps",   [])
        eval_losses  = ckpt.get("eval_losses",  [])
        lr_steps     = ckpt.get("lr_steps",     [])
        lr_values    = ckpt.get("lr_values",    [])
        print(f"  Resumed at step {start_step} → training until {AudioConfig.TRAIN_STEPS}")
        print(f"  LR = {AudioConfig.LR:.1e}")
    else:
        print("Starting from scratch.")

    # ── FIX: remaining computed from actual start_step, not hardcoded ──
    remaining = AudioConfig.TRAIN_STEPS - start_step   # = 60000 - 20000 = 40000
    print(f"  Remaining steps: {remaining:,}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(100, int(0.01 * remaining)),  # 1% warmup = 400 steps
        num_training_steps=remaining,
    )

    os.makedirs(AudioConfig.CKPT_DIR, exist_ok=True)

    progress = tqdm(initial=start_step, total=AudioConfig.TRAIN_STEPS)
    step     = start_step
    model.train()

    while step < AudioConfig.TRAIN_STEPS:
        for specs in loader:

            b            = specs.shape[0]
            t            = torch.randint(0, AudioConfig.DIFFUSION_STEPS, (b,))
            noisy, noise = sampler.add_noise(specs, t)
            pred         = model(noisy.to(device), t.to(device))
            loss         = loss_fn(pred, noise.to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            current_lr = scheduler.get_last_lr()[0]
            train_steps.append(step);  train_losses.append(loss.item())
            lr_steps.append(step);     lr_values.append(current_lr)

            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            step += 1

            if step % AudioConfig.EVAL_INTERVAL == 0:
                print(f"\n{'─'*55}")
                print(f"  Step {step:,} / {AudioConfig.TRAIN_STEPS:,}")

                eval_loss = compute_eval_loss(model, loader, sampler, loss_fn,
                                             device, n_batches=AudioConfig.EVAL_BATCHES)
                eval_steps.append(step);  eval_losses.append(eval_loss)

                gap = eval_loss / max(loss.item(), 1e-9)
                overfit_warn = "  ⚠ OVERFITTING" if gap > 2.0 else ""
                print(f"  Train loss : {loss.item():.4f}")
                print(f"  Eval  loss : {eval_loss:.4f}  (ratio {gap:.1f}x){overfit_warn}")
                print(f"  LR         : {current_lr:.6f}")

                generate_sample(model, sampler, device, step)

                ckpt_path = os.path.join(AudioConfig.CKPT_DIR, f"diffusion_{step}.pth")
                torch.save({
                    "step":         step,
                    "model":        model.state_dict(),
                    "optim":        optimizer.state_dict(),
                    "sched":        scheduler.state_dict(),
                    "loss":         loss.item(),
                    "eval_loss":    eval_loss,
                    "train_steps":  train_steps,
                    "train_losses": train_losses,
                    "eval_steps":   eval_steps,
                    "eval_losses":  eval_losses,
                    "lr_steps":     lr_steps,
                    "lr_values":    lr_values,
                }, ckpt_path)
                print(f"  Checkpoint -> {ckpt_path}")

                plot_losses(train_steps, train_losses, eval_steps, eval_losses,
                            lr_steps, lr_values, save_dir=AudioConfig.PLOTS_DIR)
                print(f"{'─'*55}")

            if step >= AudioConfig.TRAIN_STEPS:
                break

    # Final generation after loop exits
    print(f"\n{'='*55}")
    print("  Training complete — running final generation ...")
    generate_sample(model, sampler, device, step)

    final_ckpt = os.path.join(AudioConfig.CKPT_DIR, f"diffusion_final_{step}.pth")
    torch.save({
        "step":         step,
        "model":        model.state_dict(),
        "train_steps":  train_steps,
        "train_losses": train_losses,
        "eval_steps":   eval_steps,
        "eval_losses":  eval_losses,
        "lr_steps":     lr_steps,
        "lr_values":    lr_values,
    }, final_ckpt)
    print(f"  Final checkpoint -> {final_ckpt}")
    plot_losses(train_steps, train_losses, eval_steps, eval_losses,
                lr_steps, lr_values, save_dir=AudioConfig.PLOTS_DIR)
    print(f"{'='*55}\n  All done.")


if __name__ == "__main__":
    train()

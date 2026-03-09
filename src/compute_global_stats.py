"""
Compute global min/max dB values across the entire training dataset.
These values will be used for FIXED normalization so that 
spec_to_audio can perfectly reverse the mapping.

Usage:
    python compute_global_stats.py
"""
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm

# ── Must match your training config exactly ──
SAMPLE_RATE    = 24000
N_MELS         = 128
HOP_LENGTH     = 944
N_FFT          = 2048
AUDIO_SECONDS  = 5
TARGET_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS

METADATA_CSV = "/Users/sachin/Documents/diffusion/data/audio_data/metadata.csv"

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)
db_transform = T.AmplitudeToDB(top_db=80)


def main():
    df = pd.read_csv(METADATA_CSV, sep="|")
    df = df[(df.duration >= 4.0) & (df.duration <= 15.0)]
    paths = df.filepath.tolist()
    print(f"Scanning {len(paths)} audio files...")

    global_min = float("inf")
    global_max = float("-inf")

    # Also collect per-file stats for a histogram
    all_mins = []
    all_maxs = []
    errors = 0

    for path in tqdm(paths, desc="Computing dB stats"):
        try:
            wav, sr = torchaudio.load(path)

            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

            # Truncate or pad to exactly TARGET_SAMPLES
            if wav.shape[1] > TARGET_SAMPLES:
                wav = wav[:, :TARGET_SAMPLES]
            else:
                wav = torch.nn.functional.pad(wav, (0, TARGET_SAMPLES - wav.shape[1]))

            mel = mel_transform(wav)
            mel_db = db_transform(mel)

            file_min = mel_db.min().item()
            file_max = mel_db.max().item()

            all_mins.append(file_min)
            all_maxs.append(file_max)

            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Skipped {path}: {e}")

    print(f"\n{'='*55}")
    print(f"  Scanned: {len(paths) - errors} files  ({errors} errors)")
    print(f"{'='*55}")
    print(f"  GLOBAL MIN dB : {global_min:.4f}")
    print(f"  GLOBAL MAX dB : {global_max:.4f}")
    print(f"{'='*55}")

    # Percentile stats (more robust than absolute min/max)
    mins_t = torch.tensor(all_mins)
    maxs_t = torch.tensor(all_maxs)

    print(f"\n  Per-file MIN dB distribution:")
    print(f"    Mean   : {mins_t.mean():.2f}")
    print(f"    Median : {mins_t.median():.2f}")
    print(f"    1st %  : {mins_t.quantile(0.01):.2f}")
    print(f"    5th %  : {mins_t.quantile(0.05):.2f}")

    print(f"\n  Per-file MAX dB distribution:")
    print(f"    Mean   : {maxs_t.mean():.2f}")
    print(f"    Median : {maxs_t.median():.2f}")
    print(f"    95th % : {maxs_t.quantile(0.95):.2f}")
    print(f"    99th % : {maxs_t.quantile(0.99):.2f}")

    print(f"\n{'='*55}")
    print(f"  RECOMMENDED VALUES for AudioConfig:")
    print(f"    DB_MIN = {mins_t.quantile(0.01).item():.1f}")
    print(f"    DB_MAX = {maxs_t.quantile(0.99).item():.1f}")
    print(f"{'='*55}")
    print(f"\n  Use these in your audio_to_spec() and spec_to_audio():")
    print(f"    mel_db = (mel_db - DB_MIN) / (DB_MAX - DB_MIN)")
    print(f"    mel_db = mel_db * 2 - 1")
    print(f"\n  And to reverse:")
    print(f"    mel_db = (mel_norm + 1) / 2")
    print(f"    mel_db = mel_db * (DB_MAX - DB_MIN) + DB_MIN")


if __name__ == "__main__":
    main()

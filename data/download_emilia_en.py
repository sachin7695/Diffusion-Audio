import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

TARGET_HOURS = 50
TARGET_SECONDS = TARGET_HOURS * 3600

save_dir = "/Users/sachin/Documents/diffusion/data/audio_data/audio"
csv_path = "/Users/sachin/Documents/diffusion/data/audio_data/metadata.csv"

os.makedirs(save_dir, exist_ok=True)

# Load English shards
path = "Emilia/EN/*.tar"

dataset = load_dataset(
    "amphion/Emilia-Dataset",
    data_files={"en": path},
    split="en",
    streaming=True
)

# dataset = dataset.cast_column("audio", Audio())


# for i, sample in enumerate(dataset):
#     print(sample)
#     break

''' 
{
 'json': {...metadata...},
 'mp3': {
     'path': 'EN_B00000_S00000_W000000.mp3',
     'array': ...,
     'sampling_rate': 24000
 },
 'audio': None
}

'''
rows = []
total_seconds = 0
sample_id = 0

print("Starting download...")

for example in dataset:

    audio = example.get("mp3")

    if audio is None:
        continue

    audio_array = audio["array"]
    sr = audio["sampling_rate"]

    duration = len(audio_array) / sr

    if total_seconds >= TARGET_SECONDS:
        break

    filename = f"sample_{sample_id}.wav"
    filepath = os.path.join(save_dir, filename)

    sf.write(filepath, audio_array, sr)

    text = example["json"]["text"]

    rows.append({
        "filepath": filepath,
        "text": text,
        "duration": duration
    })

    total_seconds += duration
    sample_id += 1

    if sample_id % 500 == 0:
        print(f"{sample_id} clips | {total_seconds/3600:.2f} hours")

df = pd.DataFrame(rows)
df.to_csv(csv_path, sep="|", index=False)

print("Finished!")
print(f"Saved {sample_id} clips")
print(f"Total hours: {total_seconds/3600:.2f}")
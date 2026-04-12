import argparse
from pathlib import Path
import pandas as pd

#Used to Quickly Find spectogram images related to audio file


#REPLACE IF NEW PATH CREATED
MAPPING_CSV = Path(
    "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup/outputs/audio_spectrogram_mapping.csv"
)

def get_images_for_wav(wav_file: str, mapping_csv: Path = MAPPING_CSV) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv)

    # Normalize wav input to filename only
    wav_name = Path(wav_file).name

    # Keep only valid spectrogram links
    filtered = df[
        (df["Sound_file"] == wav_name) &
        (df["spectrogram_exists"].astype(str).str.lower() == "true")
    ].copy()

    return filtered.sort_values("Selection")[
        ["Selection", "Start_time", "End_time", "Call_type", "spectrogram_path"]
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_file", help="Example: 99-45_vehicle_01.wav")
    args = parser.parse_args()

    result = get_images_for_wav(args.wav_file)
    if result.empty:
        print("No spectrograms found for that wav file.")
    else:
        print(result.to_string(index=False))
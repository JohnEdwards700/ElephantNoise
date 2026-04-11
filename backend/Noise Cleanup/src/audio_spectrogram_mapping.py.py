import pandas as pd
from pathlib import Path

ANNOTATIONS_CSV = Path(
    "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/Copy of Audio Files Master (04_10_2026) - 20260324_rumbles_in_noise_for_hackathon.csv"
)
AUDIO_DIR = Path(
    "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/2026)-20260411T194629Z-3-001/Audio Files (04-10-2026)"
)
SPECTRO_DIR = Path(
    "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/2026)-20260411T194554Z-3-001/Spectro Files (04-10-2026)"
)
OUT_CSV = Path(
    "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup/outputs/audio_spectrogram_mapping.csv"
)

def find_spectrogram(base_name: str, selection: int) -> Path | None:
    expected = f"{base_name}_Selection_{selection}.png"
    p1 = SPECTRO_DIR / expected
    p2 = SPECTRO_DIR / f"{expected}:Zone.Identifier"  # if Windows ADS artifact is present
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None

def main():
    df = pd.read_csv(ANNOTATIONS_CSV)

    rows = []
    for _, r in df.iterrows():
        selection = int(r["Selection"])
        sound_file = str(r["Sound_file"])
        base_name = Path(sound_file).stem

        audio_path = AUDIO_DIR / sound_file
        spectro_path = find_spectrogram(base_name, selection)

        rows.append(
            {
                "Selection": selection,
                "Sound_file": sound_file,
                "Start_time": r.get("Start_time"),
                "End_time": r.get("End_time"),
                "Call_type": r.get("Call_type"),
                "audio_path": str(audio_path),
                "audio_exists": audio_path.exists(),
                "expected_spectrogram": f"{base_name}_Selection_{selection}.png",
                "spectrogram_path": str(spectro_path) if spectro_path else "",
                "spectrogram_exists": spectro_path is not None,
            }
        )

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Saved mapping: {OUT_CSV}")
    print(out_df[["audio_exists", "spectrogram_exists"]].value_counts().to_string())

if __name__ == "__main__":
    main()
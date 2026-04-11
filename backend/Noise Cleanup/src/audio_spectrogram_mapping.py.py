import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANNOTATIONS_CSV = PROJECT_ROOT / "data" / "annotations" / "calls.csv"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
SPECTRO_DIR = PROJECT_ROOT / "data" / "spectrogram_refs"
OUT_CSV = PROJECT_ROOT / "outputs" / "audio_spectrogram_mapping.csv"


def find_spectrogram(base_name: str, selection: int) -> Path | None:
    expected = f"{base_name}_Selection_{selection}.png"
    p1 = SPECTRO_DIR / expected
    p2 = SPECTRO_DIR / f"{expected}:Zone.Identifier"

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
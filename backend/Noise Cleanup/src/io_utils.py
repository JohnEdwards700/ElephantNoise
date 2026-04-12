"""
Input/output utilities for ElephantNoise.

This file handles:
- loading annotation and mapping CSV files
- fetching one call row by Selection
- loading WAV audio
- saving cleaned WAV audio
- ensuring output directories exist
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import soundfile as sf

from config import (
    ANNOTATIONS_CSV_PATH,
    MAPPING_CSV_PATH,
    CLEANED_AUDIO_DIR,
    ensure_output_dirs,
)


def load_annotations(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load the original annotations CSV.

    Expected columns:
    - Selection
    - Sound_file
    - Start_time
    - End_time
    - Call_type
    """
    path = csv_path if csv_path is not None else ANNOTATIONS_CSV_PATH

    if not path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["Selection", "Sound_file", "Start_time", "End_time", "Call_type"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Annotations CSV missing required columns: {missing}")

    return df


def load_mapping_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load the resolved audio/spectrogram mapping CSV.

    Expected columns:
    - Selection
    - Sound_file
    - Start_time
    - End_time
    - Call_type
    - audio_path
    - audio_exists
    - expected_spectrogram
    - spectrogram_path
    - spectrogram_exists
    """
    path = csv_path if csv_path is not None else MAPPING_CSV_PATH

    if not path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "Selection",
        "Sound_file",
        "Start_time",
        "End_time",
        "Call_type",
        "audio_path",
        "audio_exists",
        "expected_spectrogram",
        "spectrogram_path",
        "spectrogram_exists",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {missing}")

    return df


def get_call_by_selection(selection: int, mapping_df: pd.DataFrame | None = None) -> pd.Series:
    """
    Return one mapping row by Selection ID.

    Example:
        row = get_call_by_selection(1)
    """
    df = mapping_df if mapping_df is not None else load_mapping_csv()

    matches = df[df["Selection"] == selection]
    if matches.empty:
        raise ValueError(f"No row found for Selection={selection}")

    if len(matches) > 1:
        raise ValueError(f"Multiple rows found for Selection={selection}; expected unique Selection")

    return matches.iloc[0]


def get_valid_calls(mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return only rows where both the audio file and spectrogram exist.
    """
    df = mapping_df if mapping_df is not None else load_mapping_csv()

    valid_df = df[(df["audio_exists"] == True) & (df["spectrogram_exists"] == True)].copy()
    return valid_df


def load_audio(audio_path: str | Path) -> Tuple[object, int]:
    """
    Load audio from a WAV file using soundfile.

    Returns:
        y: audio samples as a NumPy array
        sr: sample rate
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, sr = sf.read(path)

    return y, sr


def save_audio(output_path: str | Path, y, sr: int) -> Path:
    """
    Save audio to disk.

    Returns:
        Path to the saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)
    return path


def build_cleaned_audio_path(selection: int, sound_file: str) -> Path:
    """
    Build a standardized cleaned-audio output path.

    Example output:
        outputs/cleaned_audio/04-040920-02_vehicle_1_selection_1_cleaned.wav
    """
    ensure_output_dirs()

    stem = Path(sound_file).stem
    filename = f"{stem}_selection_{selection}_cleaned.wav"
    return CLEANED_AUDIO_DIR / filename


def summarize_call_row(row: pd.Series) -> dict:
    """
    Convert a selected call row into a clean Python dict for downstream use.
    """
    return {
        "selection": int(row["Selection"]),
        "sound_file": str(row["Sound_file"]),
        "start_time": float(row["Start_time"]),
        "end_time": float(row["End_time"]),
        "call_type": str(row["Call_type"]),
        "audio_path": str(row["audio_path"]),
        "audio_exists": bool(row["audio_exists"]),
        "expected_spectrogram": str(row["expected_spectrogram"]),
        "spectrogram_path": str(row["spectrogram_path"]),
        "spectrogram_exists": bool(row["spectrogram_exists"]),
    }


if __name__ == "__main__":
    # Simple local test
    mapping_df = load_mapping_csv()
    valid_df = get_valid_calls(mapping_df)

    print(f"Total mapped rows: {len(mapping_df)}")
    print(f"Valid rows: {len(valid_df)}")

    if not valid_df.empty:
        first_selection = int(valid_df.iloc[0]["Selection"])
        row = get_call_by_selection(first_selection, mapping_df)
        info = summarize_call_row(row)

        print("First valid selection:")
        print(info)

        y, sr = load_audio(info["audio_path"])
        print(f"Loaded audio shape: {getattr(y, 'shape', 'unknown')}")
        print(f"Sample rate: {sr}")
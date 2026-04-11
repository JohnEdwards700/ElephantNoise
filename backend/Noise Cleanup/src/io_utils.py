"""Utilities for loading and saving audio files and CSV annotations."""

import os
import numpy as np
import soundfile as sf
import pandas as pd


def load_audio(filepath, sample_rate=None):
    """Load an audio file and return (signal, sample_rate)."""
    signal, sr = sf.read(filepath, always_2d=False)
    if sample_rate is not None and sr != sample_rate:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return signal, sr


def save_audio(filepath, signal, sample_rate):
    """Save a numpy array as an audio file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, signal, sample_rate)


def load_annotations(filepath):
    """Load call annotations from a CSV file."""
    return pd.read_csv(filepath)


def list_audio_files(directory, extensions=(".wav", ".flac", ".mp3")):
    """Return a sorted list of audio file paths in a directory."""
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(extensions)
    ]
    return files

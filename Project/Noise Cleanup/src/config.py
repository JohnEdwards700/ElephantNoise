"""
Centralized configuration for ElephantNoise.

This file stores shared constants and tunable parameters used across:
- STFT / spectrogram generation
- crop and noise window selection
- spectral subtraction
- plotting
- output naming
"""

from pathlib import Path

# =========================
# Project paths
# =========================

# This file lives in: .../backend/Noise Cleanup/src/config.py
# parents[1] -> .../backend/Noise Cleanup
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
SPECTRO_REFS_DIR = DATA_DIR / "spectrogram_refs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLEANED_AUDIO_DIR = OUTPUTS_DIR / "cleaned_audio"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MAPPING_CSV_PATH = OUTPUTS_DIR / "audio_spectrogram_mapping.csv"
ANNOTATIONS_CSV_PATH = ANNOTATIONS_DIR / "calls.csv"

# =========================
# Audio / STFT parameters
# =========================

# Use native sample rate by default unless a later step explicitly resamples.
TARGET_SAMPLE_RATE = None

# STFT settings
N_FFT = 4096
HOP_LENGTH = 512
WIN_LENGTH = 4096

# Window function used by librosa/scipy STFT functions
WINDOW = "hann"

# =========================
# Call cropping / noise estimation
# =========================

# Extra context added around the annotated elephant call
CROP_BUFFER_SEC = 0.5

# How much audio to use before and after the call for noise estimation
NOISE_BUFFER_SEC = 0.5

# Minimum allowed crop start time
MIN_TIME_SEC = 0.0

# =========================
# Spectral subtraction
# =========================

# Noise subtraction strength
ALPHA = 1.2

# Small floor to avoid overly harsh subtraction artifacts
BETA = 0.05

# Optional epsilon to avoid divide-by-zero / numerical issues
EPSILON = 1e-10

# =========================
# Plotting
# =========================

# Frequency bounds for spectrogram visualization
FMIN_PLOT = 0
FMAX_PLOT = 1000

# Figure settings
FIGSIZE = (10, 6)
DPI = 150

# =========================
# File naming
# =========================

CLEANED_AUDIO_SUFFIX = "_cleaned.wav"
BEFORE_PLOT_SUFFIX = "_before.png"
AFTER_PLOT_SUFFIX = "_after.png"
COMPARISON_PLOT_SUFFIX = "_comparison.png"

# =========================
# Utility helpers
# =========================

def ensure_output_dirs() -> None:
    """
    Create output directories if they do not already exist.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
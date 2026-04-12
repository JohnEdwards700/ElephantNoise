"""
Visualization utilities for ElephantNoise.

This file handles:
- plotting spectrograms for debugging and demo
- saving original and cleaned spectrogram images
- optionally saving before/after comparison plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

from config import (
    HOP_LENGTH,
    N_FFT,
    FMIN_PLOT,
    FMAX_PLOT,
    FIGSIZE,
    DPI,
    PLOTS_DIR,
    ensure_output_dirs,
)
from spectrogram_utils import magnitude_to_db


def _ensure_parent_dir(output_path: str | Path) -> Path:
    """
    Ensure the parent directory for an output file exists.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_spectrogram(
    mag: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    title: str = "Spectrogram",
    output_path: str | Path | None = None,
    fmin: float = FMIN_PLOT,
    fmax: float = FMAX_PLOT,
    cmap: str = "magma",
    show_colorbar: bool = True,
) -> Path | None:
    """
    Plot a single magnitude spectrogram and optionally save it.

    Args:
        mag: Magnitude spectrogram array of shape (freq_bins, time_frames)
        sr: Sample rate
        hop_length: STFT hop length
        title: Plot title
        output_path: Where to save the image. If None, the plot is not saved.
        fmin: Minimum displayed frequency
        fmax: Maximum displayed frequency
        cmap: Matplotlib colormap
        show_colorbar: Whether to display a colorbar

    Returns:
        Saved output path if output_path is provided, else None
    """
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {mag.shape}")

    mag_db = magnitude_to_db(mag)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    img = librosa.display.specshow(
        mag_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap=cmap,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylim([fmin, fmax])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if show_colorbar:
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = _ensure_parent_dir(output_path)
        fig.savefig(saved_path, bbox_inches="tight")

    plt.close(fig)
    return saved_path


def plot_before_after(
    original_mag: np.ndarray,
    cleaned_mag: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    title: str = "Before vs After Spectrogram",
    output_path: str | Path | None = None,
    fmin: float = FMIN_PLOT,
    fmax: float = FMAX_PLOT,
    cmap: str = "magma",
) -> Path | None:
    """
    Plot original and cleaned magnitude spectrograms side by side.

    Args:
        original_mag: Original magnitude spectrogram
        cleaned_mag: Cleaned magnitude spectrogram
        sr: Sample rate
        hop_length: STFT hop length
        title: Overall figure title
        output_path: Where to save the image. If None, the plot is not saved
        fmin: Minimum displayed frequency
        fmax: Maximum displayed frequency
        cmap: Matplotlib colormap

    Returns:
        Saved output path if output_path is provided, else None
    """
    if original_mag.ndim != 2:
        raise ValueError(f"Expected 2D original spectrogram, got shape {original_mag.shape}")
    if cleaned_mag.ndim != 2:
        raise ValueError(f"Expected 2D cleaned spectrogram, got shape {cleaned_mag.shape}")

    original_db = magnitude_to_db(original_mag)
    cleaned_db = magnitude_to_db(cleaned_mag)

    fig, axes = plt.subplots(2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * 1.6), dpi=DPI)

    img1 = librosa.display.specshow(
        original_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap=cmap,
        ax=axes[0],
    )
    axes[0].set_title("Original Spectrogram")
    axes[0].set_ylim([fmin, fmax])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        cleaned_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap=cmap,
        ax=axes[1],
    )
    axes[1].set_title("Cleaned Spectrogram")
    axes[1].set_ylim([fmin, fmax])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    fig.suptitle(title)
    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = _ensure_parent_dir(output_path)
        fig.savefig(saved_path, bbox_inches="tight")

    plt.close(fig)
    return saved_path


def build_plot_path(
    selection: int,
    sound_file: str,
    suffix: str,
) -> Path:
    """
    Build a standardized plot output path.

    Example:
        outputs/plots/04-040920-02_vehicle_1_selection_1_before.png
    """
    ensure_output_dirs()
    stem = Path(sound_file).stem
    filename = f"{stem}_selection_{selection}{suffix}"
    return PLOTS_DIR / filename


if __name__ == "__main__":
    # Simple local smoke test with random spectrogram-shaped data
    ensure_output_dirs()

    rng = np.random.default_rng(42)
    fake_original = np.abs(rng.normal(size=(N_FFT // 2 + 1, 120)))
    fake_cleaned = np.maximum(fake_original - 0.2, 1e-6)

    single_path = build_plot_path(
        selection=1,
        sound_file="04-040920-02_vehicle_1.wav",
        suffix="_before.png",
    )
    comparison_path = build_plot_path(
        selection=1,
        sound_file="04-040920-02_vehicle_1.wav",
        suffix="_comparison.png",
    )

    plot_spectrogram(
        mag=fake_original,
        sr=16000,
        title="Test Original Spectrogram",
        output_path=single_path,
    )

    plot_before_after(
        original_mag=fake_original,
        cleaned_mag=fake_cleaned,
        sr=16000,
        title="Test Before/After Spectrogram",
        output_path=comparison_path,
    )

    print(f"Saved single spectrogram to: {single_path}")
    print(f"Saved comparison spectrogram to: {comparison_path}")
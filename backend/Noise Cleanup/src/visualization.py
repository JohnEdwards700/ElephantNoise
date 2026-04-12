"""
Visualization utilities for ElephantNoise.

This file handles:
- plotting spectrograms for debugging and demo
- saving original and cleaned spectrogram images
- optionally saving before/after comparison plots
- drawing optional vertical guide lines for:
    - call start / end
    - pre-noise window start / end
    - post-noise window start / end
- plotting a 1D estimated noise profile
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
    call_start_time: float | None = None,
    call_end_time: float | None = None,
    pre_noise_start_time: float | None = None,
    pre_noise_end_time: float | None = None,
    post_noise_start_time: float | None = None,
    post_noise_end_time: float | None = None,
) -> Path | None:
    """
    Plot a single magnitude spectrogram and optionally save it.
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

    if call_start_time is not None:
        ax.axvline(call_start_time, linestyle="--", linewidth=2, label="Call Start")
    if call_end_time is not None:
        ax.axvline(call_end_time, linestyle="--", linewidth=2, label="Call End")

    if pre_noise_start_time is not None:
        ax.axvline(pre_noise_start_time, linestyle=":", linewidth=1.5, label="Pre-Noise Start")
    if pre_noise_end_time is not None:
        ax.axvline(pre_noise_end_time, linestyle=":", linewidth=1.5, label="Pre-Noise End")

    if post_noise_start_time is not None:
        ax.axvline(post_noise_start_time, linestyle=":", linewidth=1.5, label="Post-Noise Start")
    if post_noise_end_time is not None:
        ax.axvline(post_noise_end_time, linestyle=":", linewidth=1.5, label="Post-Noise End")

    if any(
        x is not None
        for x in [
            call_start_time,
            call_end_time,
            pre_noise_start_time,
            pre_noise_end_time,
            post_noise_start_time,
            post_noise_end_time,
        ]
    ):
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                filtered_handles.append(h)
                filtered_labels.append(l)
                seen.add(l)
        ax.legend(filtered_handles, filtered_labels, loc="upper right", fontsize=8)

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
    call_start_time: float | None = None,
    call_end_time: float | None = None,
    pre_noise_start_time: float | None = None,
    pre_noise_end_time: float | None = None,
    post_noise_start_time: float | None = None,
    post_noise_end_time: float | None = None,
) -> Path | None:
    """
    Plot original and cleaned magnitude spectrograms stacked vertically.
    """
    if original_mag.ndim != 2:
        raise ValueError(f"Expected 2D original spectrogram, got shape {original_mag.shape}")
    if cleaned_mag.ndim != 2:
        raise ValueError(f"Expected 2D cleaned spectrogram, got shape {cleaned_mag.shape}")

    original_db = magnitude_to_db(original_mag)
    cleaned_db = magnitude_to_db(cleaned_mag)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGSIZE[0], FIGSIZE[1] * 1.6),
        dpi=DPI,
    )

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

    for ax in axes:
        if call_start_time is not None:
            ax.axvline(call_start_time, linestyle="--", linewidth=2, label="Call Start")
        if call_end_time is not None:
            ax.axvline(call_end_time, linestyle="--", linewidth=2, label="Call End")

        if pre_noise_start_time is not None:
            ax.axvline(pre_noise_start_time, linestyle=":", linewidth=1.5, label="Pre-Noise Start")
        if pre_noise_end_time is not None:
            ax.axvline(pre_noise_end_time, linestyle=":", linewidth=1.5, label="Pre-Noise End")

        if post_noise_start_time is not None:
            ax.axvline(post_noise_start_time, linestyle=":", linewidth=1.5, label="Post-Noise Start")
        if post_noise_end_time is not None:
            ax.axvline(post_noise_end_time, linestyle=":", linewidth=1.5, label="Post-Noise End")

    if any(
        x is not None
        for x in [
            call_start_time,
            call_end_time,
            pre_noise_start_time,
            pre_noise_end_time,
            post_noise_start_time,
            post_noise_end_time,
        ]
    ):
        handles, labels = axes[0].get_legend_handles_labels()
        seen = set()
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                filtered_handles.append(h)
                filtered_labels.append(l)
                seen.add(l)
        axes[0].legend(filtered_handles, filtered_labels, loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = _ensure_parent_dir(output_path)
        fig.savefig(saved_path, bbox_inches="tight")

    plt.close(fig)
    return saved_path


def plot_noise_profile(
    noise_profile: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    title: str = "Estimated Noise Profile",
    output_path: str | Path | None = None,
    fmin: float = FMIN_PLOT,
    fmax: float = FMAX_PLOT,
) -> Path | None:
    """
    Plot a 1D estimated noise profile over frequency.
    """
    if noise_profile.ndim != 1:
        raise ValueError(f"Expected 1D noise profile, got shape {noise_profile.shape}")

    freqs = np.linspace(0, sr / 2, len(noise_profile))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(freqs, noise_profile, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim([fmin, fmax])
    ax.grid(True, alpha=0.3)

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
    noise_mode: str | None = None,
) -> Path:
    """
    Build a standardized plot output path.

    Example outputs:
        outputs/plots/04-040920-02_vehicle_1_selection_1_before.png
        outputs/plots/04-040920-02_vehicle_1_selection_1_post_before.png
    """
    ensure_output_dirs()
    stem = Path(sound_file).stem

    if noise_mode is None:
        filename = f"{stem}_selection_{selection}{suffix}"
    else:
        filename = f"{stem}_selection_{selection}_{noise_mode}{suffix}"

    return PLOTS_DIR / filename

def build_noise_profile_plot_path(
    selection: int,
    sound_file: str,
    noise_mode: str | None = None,
) -> Path:
    """
    Build a standardized noise-profile plot output path.
    """
    ensure_output_dirs()
    stem = Path(sound_file).stem

    if noise_mode is None:
        filename = f"{stem}_selection_{selection}_noise_profile.png"
    else:
        filename = f"{stem}_selection_{selection}_{noise_mode}_noise_profile.png"

    return PLOTS_DIR / filename


if __name__ == "__main__":
    ensure_output_dirs()

    rng = np.random.default_rng(42)
    fake_original = np.abs(rng.normal(size=(N_FFT // 2 + 1, 120)))
    fake_cleaned = np.maximum(fake_original - 0.2, 1e-6)
    fake_noise_profile = np.mean(fake_original[:, :10], axis=1)

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
    noise_profile_path = build_noise_profile_plot_path(
        selection=1,
        sound_file="04-040920-02_vehicle_1.wav",
    )

    plot_spectrogram(
        mag=fake_original,
        sr=16000,
        title="Test Original Spectrogram",
        output_path=single_path,
        call_start_time=1.0,
        call_end_time=2.5,
        pre_noise_start_time=0.5,
        pre_noise_end_time=1.0,
        post_noise_start_time=2.5,
        post_noise_end_time=3.0,
    )

    plot_before_after(
        original_mag=fake_original,
        cleaned_mag=fake_cleaned,
        sr=16000,
        title="Test Before/After Spectrogram",
        output_path=comparison_path,
        call_start_time=1.0,
        call_end_time=2.5,
        pre_noise_start_time=0.5,
        pre_noise_end_time=1.0,
        post_noise_start_time=2.5,
        post_noise_end_time=3.0,
    )

    plot_noise_profile(
        noise_profile=fake_noise_profile,
        sr=16000,
        n_fft=N_FFT,
        title="Test Noise Profile",
        output_path=noise_profile_path,
    )

    print(f"Saved single spectrogram to: {single_path}")
    print(f"Saved comparison spectrogram to: {comparison_path}")
    print(f"Saved noise profile plot to: {noise_profile_path}")
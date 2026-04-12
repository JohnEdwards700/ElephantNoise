"""
Spectral subtraction utilities for ElephantNoise.

This file handles:
- subtracting an estimated noise profile from a magnitude spectrogram
- applying a floor value to reduce harsh artifacts
- optionally smoothing the cleaned result slightly
"""

import numpy as np

from config import ALPHA, BETA, EPSILON


def apply_floor(
    clean_mag: np.ndarray,
    original_mag: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """
    Apply a spectral floor to avoid overly harsh subtraction artifacts.

    For each time-frequency bin:
        floored_mag = max(clean_mag, beta * original_mag)

    Args:
        clean_mag: cleaned magnitude spectrogram
        original_mag: original magnitude spectrogram
        beta: floor factor

    Returns:
        floored magnitude spectrogram
    """
    if clean_mag.shape != original_mag.shape:
        raise ValueError(
            f"clean_mag and original_mag must have the same shape. "
            f"Got {clean_mag.shape} and {original_mag.shape}."
        )

    floor_mag = beta * original_mag
    return np.maximum(clean_mag, floor_mag)


def smooth_spectrogram(
    mag: np.ndarray,
    kernel_size_time: int = 3,
    kernel_size_freq: int = 1,
) -> np.ndarray:
    """
    Apply simple local averaging to lightly smooth a magnitude spectrogram.

    This is optional and meant to slightly reduce patchy artifacts.
    It uses a small mean filter implemented with NumPy only.

    Args:
        mag: 2D magnitude spectrogram (freq_bins, time_frames)
        kernel_size_time: smoothing width across time
        kernel_size_freq: smoothing width across frequency

    Returns:
        Smoothed magnitude spectrogram
    """
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {mag.shape}")

    if kernel_size_time < 1 or kernel_size_freq < 1:
        raise ValueError("Kernel sizes must be >= 1")

    pad_f = kernel_size_freq // 2
    pad_t = kernel_size_time // 2

    padded = np.pad(
        mag,
        pad_width=((pad_f, pad_f), (pad_t, pad_t)),
        mode="edge",
    )

    smoothed = np.zeros_like(mag)

    for f in range(mag.shape[0]):
        for t in range(mag.shape[1]):
            window = padded[
                f : f + kernel_size_freq,
                t : t + kernel_size_time,
            ]
            smoothed[f, t] = np.mean(window)

    return smoothed


def spectral_subtract(
    mag: np.ndarray,
    noise_profile: np.ndarray,
    call_frame_start: int | None = None,
    call_frame_end: int | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    smooth: bool = False,
) -> np.ndarray:
    """
    Apply spectral subtraction in the magnitude domain.

    Core formula:
        clean_mag = max(noisy_mag - alpha * noise_profile, beta * noisy_mag)

    Args:
        mag: original magnitude spectrogram of shape (freq_bins, time_frames)
        noise_profile: 1D noise magnitude estimate of shape (freq_bins,)
        call_frame_start: optional start frame of the call region
        call_frame_end: optional end frame of the call region
        alpha: subtraction strength
        beta: floor factor
        smooth: whether to lightly smooth the cleaned result

    Returns:
        cleaned magnitude spectrogram
    """
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {mag.shape}")

    if noise_profile.ndim != 1:
        raise ValueError(f"Expected 1D noise profile, got shape {noise_profile.shape}")

    n_freqs, n_frames = mag.shape

    if noise_profile.shape[0] != n_freqs:
        raise ValueError(
            f"Noise profile length must match spectrogram frequency bins. "
            f"Got {noise_profile.shape[0]} and {n_freqs}."
        )

    # Default: apply subtraction to the full spectrogram
    if call_frame_start is None:
        call_frame_start = 0
    if call_frame_end is None:
        call_frame_end = n_frames

    call_frame_start = max(0, min(call_frame_start, n_frames))
    call_frame_end = max(0, min(call_frame_end, n_frames))

    if call_frame_end <= call_frame_start:
        raise ValueError(
            f"Invalid call frame range: start={call_frame_start}, end={call_frame_end}"
        )

    cleaned_mag = mag.copy()

    # Expand noise profile to match the selected time range
    noise_matrix = noise_profile[:, np.newaxis]

    target_region = mag[:, call_frame_start:call_frame_end]

    # Raw subtraction
    subtracted = target_region - alpha * noise_matrix

    # Prevent negatives
    subtracted = np.maximum(subtracted, 0.0)

    # Apply floor
    floored = apply_floor(subtracted, target_region, beta=beta)

    # Optional light smoothing
    if smooth:
        floored = smooth_spectrogram(floored, kernel_size_time=3, kernel_size_freq=1)

    cleaned_mag[:, call_frame_start:call_frame_end] = floored

    # Numerical safety
    cleaned_mag = np.maximum(cleaned_mag, EPSILON)

    return cleaned_mag


if __name__ == "__main__":
    # Simple smoke test
    rng = np.random.default_rng(42)

    # Fake magnitude spectrogram: (freq_bins, time_frames)
    mag = np.abs(rng.normal(size=(257, 100)))

    # Fake noise profile
    noise_profile = np.mean(mag[:, :10], axis=1)

    cleaned = spectral_subtract(
        mag=mag,
        noise_profile=noise_profile,
        call_frame_start=20,
        call_frame_end=80,
        alpha=1.2,
        beta=0.05,
        smooth=True,
    )

    print("Original shape:", mag.shape)
    print("Noise profile shape:", noise_profile.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Min cleaned value:", cleaned.min())
    print("Max cleaned value:", cleaned.max())
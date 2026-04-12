"""
Spectral subtraction utilities for ElephantNoise.

This file handles:
- subtracting an estimated noise profile from a magnitude spectrogram
- applying a floor value to reduce harsh artifacts
- optionally smoothing the cleaned result slightly
- tapering subtraction strength around a call region
- applying weaker subtraction below a chosen frequency threshold
"""

import numpy as np

from .config import ALPHA, BETA, EPSILON


def apply_floor(
    clean_mag: np.ndarray,
    original_mag: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """
    Apply a spectral floor to avoid overly harsh subtraction artifacts.

    For each time-frequency bin:
        floored_mag = max(clean_mag, beta * original_mag)
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


def build_time_weight_mask(
    n_frames: int,
    call_frame_start: int | None = None,
    call_frame_end: int | None = None,
    transition_frames: int = 8,
) -> np.ndarray:
    """
    Build a soft time weighting mask for subtraction strength.

    Returns a 1D vector of length n_frames with values in [0, 1].
    """
    weights = np.ones(n_frames, dtype=float)

    if call_frame_start is None or call_frame_end is None:
        return weights

    call_frame_start = max(0, min(call_frame_start, n_frames))
    call_frame_end = max(0, min(call_frame_end, n_frames))

    if call_frame_end <= call_frame_start:
        return weights

    # Gentle subtraction outside the call
    weights[:] = 0.2

    # Strongest subtraction inside the call
    weights[call_frame_start:call_frame_end] = 1.0

    if transition_frames > 0:
        left_start = max(0, call_frame_start - transition_frames)
        left_end = call_frame_start
        if left_end > left_start:
            ramp = np.linspace(0.35, 1.0, left_end - left_start, endpoint=False)
            weights[left_start:left_end] = ramp

        right_start = call_frame_end
        right_end = min(n_frames, call_frame_end + transition_frames)
        if right_end > right_start:
            ramp = np.linspace(1.0, 0.35, right_end - right_start, endpoint=False)
            weights[right_start:right_end] = ramp

    return weights


def build_frequency_weight_profile(
    n_freqs: int,
    sr: int,
    low_freq_threshold_hz: float = 120.0,
    mid_freq_threshold_hz: float = 250.0,
    low_freq_scale: float = 0.4,
    mid_freq_scale: float = 0.7,
) -> np.ndarray:
    """
    Build a frequency-dependent weighting profile for the noise subtraction.

    Idea:
    - Below low_freq_threshold_hz: protect the rumble band more
    - Between low_freq_threshold_hz and mid_freq_threshold_hz: moderate subtraction
    - Above mid_freq_threshold_hz: normal subtraction

    Returns:
        1D vector of shape (n_freqs,)
    """
    freqs = np.linspace(0, sr / 2, n_freqs)
    freq_weights = np.ones(n_freqs, dtype=float)

    freq_weights[freqs < low_freq_threshold_hz] = low_freq_scale

    mid_mask = (freqs >= low_freq_threshold_hz) & (freqs < mid_freq_threshold_hz)
    freq_weights[mid_mask] = mid_freq_scale

    return freq_weights


def spectral_subtract(
    mag: np.ndarray,
    noise_profile: np.ndarray,
    call_frame_start: int | None = None,
    call_frame_end: int | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    smooth: bool = False,
    use_soft_time_mask: bool = True,
    transition_frames: int = 8,
    sr: int = 2000,
    use_frequency_weighting: bool = True,
    low_freq_threshold_hz: float = 120.0,
    mid_freq_threshold_hz: float = 250.0,
    low_freq_scale: float = 0.4,
    mid_freq_scale: float = 0.7,
) -> np.ndarray:
    """
    Apply spectral subtraction in the magnitude domain.

    Core formula:
        clean_mag = max(noisy_mag - alpha * time_weight(t) * freq_weight(f) * noise_profile,
                        beta * noisy_mag)

    Args:
        mag: original magnitude spectrogram of shape (freq_bins, time_frames)
        noise_profile: 1D noise magnitude estimate of shape (freq_bins,)
        call_frame_start: optional start frame of the call region
        call_frame_end: optional end frame of the call region
        alpha: subtraction strength
        beta: floor factor
        smooth: whether to lightly smooth the cleaned result
        use_soft_time_mask: whether to taper subtraction around the call region
        transition_frames: number of frames used to ramp subtraction strength
        sr: sample rate for frequency-axis weighting
        use_frequency_weighting: whether to protect low frequencies
        low_freq_threshold_hz: below this, subtraction is reduced strongly
        mid_freq_threshold_hz: between low and mid thresholds, subtraction is reduced moderately
        low_freq_scale: scaling factor below low_freq_threshold_hz
        mid_freq_scale: scaling factor between thresholds

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

    if use_soft_time_mask:
        time_weights = build_time_weight_mask(
            n_frames=n_frames,
            call_frame_start=call_frame_start,
            call_frame_end=call_frame_end,
            transition_frames=transition_frames,
        )
    else:
        time_weights = np.ones(n_frames, dtype=float)

    if use_frequency_weighting:
        freq_weights = build_frequency_weight_profile(
            n_freqs=n_freqs,
            sr=sr,
            low_freq_threshold_hz=low_freq_threshold_hz,
            mid_freq_threshold_hz=mid_freq_threshold_hz,
            low_freq_scale=low_freq_scale,
            mid_freq_scale=mid_freq_scale,
        )
    else:
        freq_weights = np.ones(n_freqs, dtype=float)

    # Expand dimensions for broadcasting
    noise_matrix = noise_profile[:, np.newaxis]
    freq_weight_matrix = freq_weights[:, np.newaxis]
    time_weight_matrix = time_weights[np.newaxis, :]

    weighted_noise = alpha * noise_matrix * freq_weight_matrix * time_weight_matrix

    # Raw subtraction across the full crop
    subtracted = mag - weighted_noise

    # Prevent negatives
    subtracted = np.maximum(subtracted, 0.0)

    # Apply floor against the original full spectrogram
    floored = apply_floor(subtracted, mag, beta=beta)

    # Optional smoothing
    if smooth:
        floored = smooth_spectrogram(floored, kernel_size_time=3, kernel_size_freq=1)

    cleaned_mag = np.maximum(floored, EPSILON)
    return cleaned_mag


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    mag = np.abs(rng.normal(size=(257, 100)))
    noise_profile = np.mean(mag[:, :10], axis=1)

    cleaned = spectral_subtract(
        mag=mag,
        noise_profile=noise_profile,
        call_frame_start=20,
        call_frame_end=80,
        alpha=0.9,
        beta=0.1,
        smooth=True,
        use_soft_time_mask=True,
        transition_frames=8,
        sr=2000,
        use_frequency_weighting=True,
        low_freq_threshold_hz=120.0,
        mid_freq_threshold_hz=250.0,
        low_freq_scale=0.4,
        mid_freq_scale=0.7,
    )

    print("Original shape:", mag.shape)
    print("Noise profile shape:", noise_profile.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Min cleaned value:", cleaned.min())
    print("Max cleaned value:", cleaned.max())

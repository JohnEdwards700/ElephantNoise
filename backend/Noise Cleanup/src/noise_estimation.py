"""
Noise estimation utilities for ElephantNoise.

This file handles:
- selecting pre-call and post-call noise windows
- converting those windows into STFT frame ranges
- estimating an average frequency-wise noise profile
- allowing selection of pre-only, post-only, or both windows
"""

from typing import Tuple

import numpy as np

from .config import NOISE_BUFFER_SEC, EPSILON
from .spectrogram_utils import time_to_sample, time_to_frame


def get_noise_windows(
    y: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    buffer_sec: float = NOISE_BUFFER_SEC,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Return sample-index windows for the pre-call and post-call noise regions.
    """
    n_samples = len(y)

    call_start_sample = time_to_sample(start_time, sr)
    call_end_sample = time_to_sample(end_time, sr)

    pre_start = max(0, call_start_sample - time_to_sample(buffer_sec, sr))
    pre_end = max(0, call_start_sample)

    post_start = min(n_samples, call_end_sample)
    post_end = min(n_samples, call_end_sample + time_to_sample(buffer_sec, sr))

    return (pre_start, pre_end), (post_start, post_end)


def get_noise_frame_ranges(
    sr: int,
    start_time: float,
    end_time: float,
    buffer_sec: float = NOISE_BUFFER_SEC,
    hop_length: int = 512,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Return STFT frame-index ranges for the pre-call and post-call noise regions.
    """
    pre_start_time = max(0.0, start_time - buffer_sec)
    pre_end_time = max(0.0, start_time)

    post_start_time = max(0.0, end_time)
    post_end_time = max(post_start_time, end_time + buffer_sec)

    pre_start_frame = time_to_frame(pre_start_time, sr, hop_length)
    pre_end_frame = time_to_frame(pre_end_time, sr, hop_length)

    post_start_frame = time_to_frame(post_start_time, sr, hop_length)
    post_end_frame = time_to_frame(post_end_time, sr, hop_length)

    return (pre_start_frame, pre_end_frame), (post_start_frame, post_end_frame)


def estimate_noise_profile(
    mag_spectrogram: np.ndarray,
    call_frame_start: int,
    call_frame_end: int,
) -> np.ndarray:
    """
    Estimate a frequency-wise noise profile from frames outside the call region.
    """
    if mag_spectrogram.ndim != 2:
        raise ValueError(
            f"Expected 2D magnitude spectrogram, got shape {mag_spectrogram.shape}"
        )

    n_freqs, n_frames = mag_spectrogram.shape

    call_frame_start = max(0, min(call_frame_start, n_frames))
    call_frame_end = max(0, min(call_frame_end, n_frames))

    if call_frame_end <= call_frame_start:
        raise ValueError(
            f"Invalid call frame range: start={call_frame_start}, end={call_frame_end}"
        )

    pre_frames = mag_spectrogram[:, :call_frame_start]
    post_frames = mag_spectrogram[:, call_frame_end:]

    available_regions = []
    if pre_frames.shape[1] > 0:
        available_regions.append(pre_frames)
    if post_frames.shape[1] > 0:
        available_regions.append(post_frames)

    if not available_regions:
        raise ValueError("No noise-only frames available before or after the call region.")

    noise_frames = np.concatenate(available_regions, axis=1)
    noise_profile = np.mean(noise_frames, axis=1)
    noise_profile = np.maximum(noise_profile, EPSILON)

    if noise_profile.shape != (n_freqs,):
        raise ValueError(
            f"Unexpected noise profile shape: {noise_profile.shape}, expected ({n_freqs},)"
        )

    return noise_profile


def estimate_noise_profile_from_frame_ranges(
    mag_spectrogram: np.ndarray,
    pre_frame_range: Tuple[int, int],
    post_frame_range: Tuple[int, int],
    noise_mode: str = "both",
) -> np.ndarray:
    """
    Estimate a frequency-wise noise profile using explicit pre/post frame ranges.

    Args:
        mag_spectrogram: magnitude spectrogram of shape (freq_bins, time_frames)
        pre_frame_range: (start_frame, end_frame) before the call
        post_frame_range: (start_frame, end_frame) after the call
        noise_mode: one of {"pre", "post", "both"}

    Returns:
        noise_profile: 1D array of shape (freq_bins,)
    """
    if mag_spectrogram.ndim != 2:
        raise ValueError(
            f"Expected 2D magnitude spectrogram, got shape {mag_spectrogram.shape}"
        )

    if noise_mode not in {"pre", "post", "both"}:
        raise ValueError(
            f"Invalid noise_mode='{noise_mode}'. Expected one of: 'pre', 'post', 'both'"
        )

    n_freqs, n_frames = mag_spectrogram.shape

    pre_start, pre_end = pre_frame_range
    post_start, post_end = post_frame_range

    pre_start = max(0, min(pre_start, n_frames))
    pre_end = max(0, min(pre_end, n_frames))
    post_start = max(0, min(post_start, n_frames))
    post_end = max(0, min(post_end, n_frames))

    available_regions = []

    if noise_mode in {"pre", "both"} and pre_end > pre_start:
        available_regions.append(mag_spectrogram[:, pre_start:pre_end])

    if noise_mode in {"post", "both"} and post_end > post_start:
        available_regions.append(mag_spectrogram[:, post_start:post_end])

    if not available_regions:
        raise ValueError(
            f"No valid noise frames available for noise_mode='{noise_mode}'."
        )

    noise_frames = np.concatenate(available_regions, axis=1)
    noise_profile = np.mean(noise_frames, axis=1)
    noise_profile = np.maximum(noise_profile, EPSILON)

    if noise_profile.shape != (n_freqs,):
        raise ValueError(
            f"Unexpected noise profile shape: {noise_profile.shape}, expected ({n_freqs},)"
        )

    return noise_profile


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    mag = np.abs(rng.normal(size=(257, 100)))
    pre_range = (10, 20)
    post_range = (80, 95)

    for mode in ["pre", "post", "both"]:
        profile = estimate_noise_profile_from_frame_ranges(
            mag_spectrogram=mag,
            pre_frame_range=pre_range,
            post_frame_range=post_range,
            noise_mode=mode,
        )
        print(f"Mode={mode}, profile shape={profile.shape}, first 5={profile[:5]}")

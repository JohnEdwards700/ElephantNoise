"""Noise profile estimation from audio magnitude spectrograms."""

import numpy as np


def estimate_noise_profile(magnitude, percentile=10):
    """Estimate the noise floor as a per-frequency percentile of the magnitude.

    Parameters
    ----------
    magnitude : np.ndarray, shape (n_freqs, n_frames)
        Magnitude spectrogram.
    percentile : int
        Percentile along the time axis used to estimate the noise floor.

    Returns
    -------
    noise_profile : np.ndarray, shape (n_freqs,)
        Estimated noise magnitude per frequency bin.
    """
    noise_profile = np.percentile(magnitude, percentile, axis=1)
    return noise_profile


def smooth_noise_profile(noise_profile, window_size=5):
    """Apply a simple moving-average smoothing to the noise profile."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(noise_profile, kernel, mode="same")

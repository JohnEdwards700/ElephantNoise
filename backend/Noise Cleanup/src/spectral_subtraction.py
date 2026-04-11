"""Spectral subtraction for noise reduction."""

import numpy as np


def spectral_subtraction(magnitude, noise_profile, over_subtraction=1.0, spectral_floor=0.002):
    """Apply spectral subtraction to a magnitude spectrogram.

    Parameters
    ----------
    magnitude : np.ndarray, shape (n_freqs, n_frames)
        Magnitude spectrogram of the noisy signal.
    noise_profile : np.ndarray, shape (n_freqs,)
        Estimated noise magnitude per frequency bin.
    over_subtraction : float
        Over-subtraction factor (alpha). Values > 1 increase noise reduction
        at the cost of potential musical noise artefacts.
    spectral_floor : float
        Minimum magnitude value after subtraction to avoid negative values.

    Returns
    -------
    cleaned_magnitude : np.ndarray, shape (n_freqs, n_frames)
        Noise-reduced magnitude spectrogram.
    """
    noise = noise_profile[:, np.newaxis] * over_subtraction
    cleaned = magnitude - noise
    cleaned = np.maximum(cleaned, spectral_floor * magnitude)
    return cleaned

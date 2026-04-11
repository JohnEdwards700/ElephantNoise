"""Utilities for computing and manipulating spectrograms."""

import numpy as np
import librosa


def compute_stft(signal, n_fft=2048, hop_length=512):
    """Compute the Short-Time Fourier Transform of a signal.

    Returns the complex STFT matrix.
    """
    return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)


def stft_to_magnitude_phase(stft_matrix):
    """Separate a complex STFT into magnitude and phase components."""
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    return magnitude, phase


def reconstruct_signal(magnitude, phase, hop_length=512):
    """Reconstruct a time-domain signal from magnitude and phase."""
    complex_stft = magnitude * np.exp(1j * phase)
    return librosa.istft(complex_stft, hop_length=hop_length)


def power_to_db(magnitude, ref=1.0, amin=1e-10):
    """Convert a magnitude spectrogram to decibels."""
    return librosa.amplitude_to_db(magnitude, ref=ref, amin=amin)

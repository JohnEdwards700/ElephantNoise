"""Visualization utilities for spectrograms and noise analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa


def plot_spectrogram_comparison(original_magnitude, cleaned_magnitude, sample_rate, save_path, hop_length=512):
    """Plot original and cleaned spectrograms side by side and save to disk.

    Parameters
    ----------
    original_magnitude : np.ndarray
        Magnitude spectrogram of the original (noisy) signal.
    cleaned_magnitude : np.ndarray
        Magnitude spectrogram of the cleaned signal.
    sample_rate : int
        Sample rate of the audio.
    save_path : str
        File path where the plot image will be saved.
    hop_length : int
        Hop length used during STFT computation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, magnitude, title in zip(
        axes,
        [original_magnitude, cleaned_magnitude],
        ["Original (Noisy)", "Cleaned"],
    ):
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        img = ax.imshow(
            db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, magnitude.shape[1] * hop_length / sample_rate, 0, sample_rate / 2],
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_noise_profile(noise_profile, sample_rate, n_fft=2048, save_path=None):
    """Plot the estimated noise profile across frequency bins.

    Parameters
    ----------
    noise_profile : np.ndarray, shape (n_freqs,)
        Estimated noise magnitude per frequency bin.
    sample_rate : int
        Sample rate of the audio.
    n_fft : int
        FFT window size.
    save_path : str or None
        If provided, the plot is saved to this path; otherwise it is shown.
    """
    freqs = np.linspace(0, sample_rate / 2, len(noise_profile))
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, noise_profile)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Estimated Noise Profile")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

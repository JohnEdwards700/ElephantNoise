"""
Spectrogram / STFT utilities for ElephantNoise.

This file handles:
- computing STFT
- splitting STFT into magnitude and phase
- reconstructing audio from magnitude + phase
- converting time to sample/frame indices
- converting frames back to time
- converting magnitude to decibels for plotting
"""

from typing import Tuple

import numpy as np
import librosa

from .config import (
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    WINDOW,
    EPSILON,
)


def time_to_sample(time_sec: float, sr: int) -> int:
    """
    Convert time in seconds to sample index.
    """
    return int(round(time_sec * sr))


def sample_to_time(sample_idx: int, sr: int) -> float:
    """
    Convert sample index to time in seconds.
    """
    return sample_idx / sr


def time_to_frame(time_sec: float, sr: int, hop_length: int = HOP_LENGTH) -> int:
    """
    Convert time in seconds to STFT frame index.
    """
    sample_idx = time_to_sample(time_sec, sr)
    return int(round(sample_idx / hop_length))


def frame_to_time(frame_idx: int, sr: int, hop_length: int = HOP_LENGTH) -> float:
    """
    Convert STFT frame index to time in seconds.
    """
    sample_idx = frame_idx * hop_length
    return sample_to_time(sample_idx, sr)


def crop_audio(
    y: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Crop audio between start_time and end_time.
    """
    start_sample = max(0, time_to_sample(start_time, sr))
    end_sample = min(len(y), time_to_sample(end_time, sr))

    if end_sample <= start_sample:
        raise ValueError(
            f"Invalid crop range: start_time={start_time}, end_time={end_time}"
        )

    return y[start_sample:end_sample]


def compute_stft(
    y: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
) -> np.ndarray:
    """
    Compute complex STFT from waveform.
    """
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def stft_to_mag_phase(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split complex STFT into magnitude and phase.

    Returns:
        magnitude: non-negative real-valued spectrogram
        phase: unit-magnitude complex phase
    """
    magnitude, phase = librosa.magphase(D)
    return magnitude, phase


def mag_phase_to_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Recombine magnitude and phase into a complex STFT matrix.
    """
    if magnitude.shape != phase.shape:
        raise ValueError(
            f"Magnitude and phase must have the same shape. "
            f"Got {magnitude.shape} and {phase.shape}."
        )

    return magnitude * phase


def reconstruct_from_mag_phase(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    length: int | None = None,
) -> np.ndarray:
    """
    Reconstruct waveform from magnitude + phase using inverse STFT.
    """
    D = mag_phase_to_complex(magnitude, phase)

    y = librosa.istft(
        stft_matrix=D,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )
    return y


def magnitude_to_db(magnitude: np.ndarray, ref: float | None = None) -> np.ndarray:
    """
    Convert magnitude spectrogram to decibel scale for plotting.

    If ref is None, uses the max magnitude as reference.
    """
    magnitude = np.maximum(magnitude, EPSILON)

    if ref is None:
        ref = float(np.max(magnitude))

    return librosa.amplitude_to_db(magnitude, ref=ref)


def power_to_db(power: np.ndarray, ref: float | None = None) -> np.ndarray:
    """
    Convert power spectrogram to decibel scale for plotting.
    """
    power = np.maximum(power, EPSILON)

    if ref is None:
        ref = float(np.max(power))

    return librosa.power_to_db(power, ref=ref)


def magnitude_to_power(magnitude: np.ndarray) -> np.ndarray:
    """
    Convert magnitude spectrogram to power spectrogram.
    """
    return magnitude ** 2


def get_frequency_bins(sr: int, n_fft: int = N_FFT) -> np.ndarray:
    """
    Get frequency values for STFT rows.
    """
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft)


def get_time_bins(
    num_frames: int,
    sr: int,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Get time values for STFT columns.
    """
    return librosa.frames_to_time(
        frames=np.arange(num_frames),
        sr=sr,
        hop_length=hop_length,
    )


if __name__ == "__main__":
    # Simple local smoke test
    import numpy as np

    sr = 16000
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # Synthetic test signal: two sine waves
    y = 0.6 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)

    D = compute_stft(y)
    mag, phase = stft_to_mag_phase(D)
    y_reconstructed = reconstruct_from_mag_phase(mag, phase, length=len(y))

    print("Original shape:", y.shape)
    print("STFT shape:", D.shape)
    print("Magnitude shape:", mag.shape)
    print("Phase shape:", phase.shape)
    print("Reconstructed shape:", y_reconstructed.shape)

    reconstruction_error = np.mean(np.abs(y - y_reconstructed))
    print("Mean absolute reconstruction error:", reconstruction_error)

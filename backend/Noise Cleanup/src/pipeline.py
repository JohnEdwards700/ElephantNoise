"""End-to-end noise cleanup pipeline."""

import os
from . import config
from .io_utils import load_audio, save_audio, list_audio_files
from .spectrogram_utils import compute_stft, stft_to_magnitude_phase, reconstruct_signal
from .noise_estimation import estimate_noise_profile, smooth_noise_profile
from .spectral_subtraction import spectral_subtraction
from .visualization import plot_spectrogram_comparison


def process_file(input_path, output_dir, plots_dir):
    """Run the noise cleanup pipeline on a single audio file."""
    basename = os.path.splitext(os.path.basename(input_path))[0]

    signal, sr = load_audio(input_path, sample_rate=config.SAMPLE_RATE)

    stft = compute_stft(signal, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    magnitude, phase = stft_to_magnitude_phase(stft)

    noise_profile = estimate_noise_profile(magnitude, percentile=config.NOISE_PERCENTILE)
    noise_profile = smooth_noise_profile(noise_profile)

    cleaned_magnitude = spectral_subtraction(
        magnitude,
        noise_profile,
        over_subtraction=config.OVER_SUBTRACTION_FACTOR,
        spectral_floor=config.SPECTRAL_FLOOR,
    )

    cleaned_signal = reconstruct_signal(cleaned_magnitude, phase, hop_length=config.HOP_LENGTH)

    output_path = os.path.join(output_dir, basename + "_cleaned.wav")
    save_audio(output_path, cleaned_signal, sr)

    plot_path = os.path.join(plots_dir, basename + "_spectrogram.png")
    plot_spectrogram_comparison(magnitude, cleaned_magnitude, sr, plot_path)

    return output_path


def run_pipeline():
    """Process all raw audio files through the noise cleanup pipeline."""
    audio_files = list_audio_files(config.RAW_AUDIO_DIR)
    if not audio_files:
        print("No audio files found in", config.RAW_AUDIO_DIR)
        return

    os.makedirs(config.CLEANED_AUDIO_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    for filepath in audio_files:
        print(f"Processing: {filepath}")
        output_path = process_file(filepath, config.CLEANED_AUDIO_DIR, config.PLOTS_DIR)
        print(f"  Saved cleaned audio to: {output_path}")

"""
End-to-end processing pipeline for ElephantNoise.

This file orchestrates:
- loading one selected call from the mapping CSV
- loading the corresponding WAV audio
- cropping around the annotated call
- computing the STFT / spectrogram
- estimating background noise
- applying spectral subtraction
- reconstructing cleaned audio
- saving cleaned WAV and spectrogram plots
- returning metadata + output paths
"""

from pathlib import Path

from config import (
    CROP_BUFFER_SEC,
    HOP_LENGTH,
    WIN_LENGTH,
    WINDOW,
    N_FFT,
    ALPHA,
    BETA,
    BEFORE_PLOT_SUFFIX,
    AFTER_PLOT_SUFFIX,
    COMPARISON_PLOT_SUFFIX,
    ensure_output_dirs,
)
from io_utils import (
    load_mapping_csv,
    get_call_by_selection,
    load_audio,
    save_audio,
    build_cleaned_audio_path,
    summarize_call_row,
)
from spectrogram_utils import (
    crop_audio,
    compute_stft,
    stft_to_mag_phase,
    reconstruct_from_mag_phase,
    time_to_frame,
)
from noise_estimation import (
    get_noise_frame_ranges,
    estimate_noise_profile_from_frame_ranges,
)
from spectral_subtraction import spectral_subtract
from visualization import plot_spectrogram, plot_before_after, build_plot_path


def process_call(selection: int) -> dict:
    """
    Run the full denoising pipeline for one selected call.

    Args:
        selection: unique Selection ID from the mapping CSV

    Returns:
        dict with metadata and output file paths
    """
    ensure_output_dirs()

    # 1. Load mapping CSV and selected row
    mapping_df = load_mapping_csv()
    row = get_call_by_selection(selection, mapping_df)
    info = summarize_call_row(row)

    sound_file = info["sound_file"]
    start_time = info["start_time"]
    end_time = info["end_time"]
    call_type = info["call_type"]
    audio_path = info["audio_path"]

    if not info["audio_exists"]:
        raise FileNotFoundError(f"Mapped audio does not exist for selection {selection}: {audio_path}")

    # 2. Load full WAV audio
    y_full, sr = load_audio(audio_path)

    # 3. Crop around the annotated call with a small buffer
    crop_start = max(0.0, start_time - CROP_BUFFER_SEC)
    crop_end = end_time + CROP_BUFFER_SEC

    y_crop = crop_audio(
        y=y_full,
        sr=sr,
        start_time=crop_start,
        end_time=crop_end,
    )

    # 4. Compute STFT on cropped audio
    D = compute_stft(
        y=y_crop,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
    )
    original_mag, phase = stft_to_mag_phase(D)

    # 5. Convert original call times into crop-relative times
    call_start_in_crop = start_time - crop_start
    call_end_in_crop = end_time - crop_start

    # 6. Estimate noise using pre-call and post-call frame ranges
    pre_frames, post_frames = get_noise_frame_ranges(
        sr=sr,
        start_time=call_start_in_crop,
        end_time=call_end_in_crop,
        buffer_sec=CROP_BUFFER_SEC,
        hop_length=HOP_LENGTH,
    )

    noise_profile = estimate_noise_profile_from_frame_ranges(
        mag_spectrogram=original_mag,
        pre_frame_range=pre_frames,
        post_frame_range=post_frames,
    )

    # 7. Identify call frame range within the cropped spectrogram
    call_frame_start = time_to_frame(call_start_in_crop, sr, HOP_LENGTH)
    call_frame_end = time_to_frame(call_end_in_crop, sr, HOP_LENGTH)

    # Guard against edge cases
    call_frame_start = max(0, min(call_frame_start, original_mag.shape[1]))
    call_frame_end = max(call_frame_start + 1, min(call_frame_end, original_mag.shape[1]))

    # 8. Apply spectral subtraction
    cleaned_mag = spectral_subtract(
        mag=original_mag,
        noise_profile=noise_profile,
        call_frame_start=call_frame_start,
        call_frame_end=call_frame_end,
        alpha=ALPHA,
        beta=BETA,
    )

    # 9. Reconstruct cleaned waveform
    y_clean = reconstruct_from_mag_phase(
        magnitude=cleaned_mag,
        phase=phase,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        length=len(y_crop),
    )

    # 10. Save cleaned audio
    cleaned_audio_path = build_cleaned_audio_path(selection=selection, sound_file=sound_file)
    save_audio(cleaned_audio_path, y_clean, sr)

    # 11. Save plots
    before_plot_path = build_plot_path(
        selection=selection,
        sound_file=sound_file,
        suffix=BEFORE_PLOT_SUFFIX,
    )
    after_plot_path = build_plot_path(
        selection=selection,
        sound_file=sound_file,
        suffix=AFTER_PLOT_SUFFIX,
    )
    comparison_plot_path = build_plot_path(
        selection=selection,
        sound_file=sound_file,
        suffix=COMPARISON_PLOT_SUFFIX,
    )

    plot_spectrogram(
        mag=original_mag,
        sr=sr,
        hop_length=HOP_LENGTH,
        title=f"Original Spectrogram | Selection {selection}",
        output_path=before_plot_path,
    )

    plot_spectrogram(
        mag=cleaned_mag,
        sr=sr,
        hop_length=HOP_LENGTH,
        title=f"Cleaned Spectrogram | Selection {selection}",
        output_path=after_plot_path,
    )

    plot_before_after(
        original_mag=original_mag,
        cleaned_mag=cleaned_mag,
        sr=sr,
        hop_length=HOP_LENGTH,
        title=f"Before vs After | Selection {selection}",
        output_path=comparison_plot_path,
    )

    # 12. Return result dictionary
    return {
        "selection": selection,
        "sound_file": sound_file,
        "start_time": start_time,
        "end_time": end_time,
        "call_type": call_type,
        "audio_path": audio_path,
        "cleaned_audio_path": str(cleaned_audio_path),
        "before_plot_path": str(before_plot_path),
        "after_plot_path": str(after_plot_path),
        "comparison_plot_path": str(comparison_plot_path),
        "sample_rate": sr,
        "crop_start": crop_start,
        "crop_end": crop_end,
        "call_frame_start": call_frame_start,
        "call_frame_end": call_frame_end,
    }


if __name__ == "__main__":
    # Simple local smoke test
    test_selection = 1
    result = process_call(test_selection)

    print("Pipeline completed successfully.")
    for key, value in result.items():
        print(f"{key}: {value}")
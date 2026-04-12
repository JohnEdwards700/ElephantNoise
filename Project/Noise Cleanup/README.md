# Elephant Noise Cleanup

This project cleans annotated elephant-call recordings by estimating background noise around each call and subtracting that noise from the call segment. It also generates spectrogram plots so you can inspect what changed before and after cleanup.

The repository includes two main experiences:

- `app.py` provides a Streamlit interface for exploring examples and viewing outputs.
- `src/pipeline.py` runs the denoising pipeline for one annotated selection at a time.

## Purpose

Field recordings often contain aircraft, vehicles, generators, and other environmental noise that can mask an elephant call. This project is designed to:

- isolate the annotated call with a small time buffer
- estimate surrounding background noise from the audio just before and/or after the call
- apply magnitude-domain spectral subtraction
- reconstruct a cleaned waveform
- save visual artifacts that make the cleanup easy to inspect

In practice, the result is not "perfect source separation." The goal is a cleaner version of the selected call region that reduces steady background noise while preserving the call structure as much as possible.

## What The Pipeline Produces

For each processed call selection, the pipeline writes:

- a cleaned WAV file
- an original spectrogram image
- a cleaned spectrogram image
- a stacked before/after comparison plot
- a 1D noise-profile plot

These files are written under:

- `outputs/cleaned_audio/`
- `outputs/plots/`

Example output naming:

```text
outputs/cleaned_audio/99-45_airplane_01_selection_1_pre_cleaned.wav
outputs/cleaned_audio/99-45_airplane_01_selection_1_post_cleaned.wav
outputs/cleaned_audio/99-45_airplane_01_selection_1_both_cleaned.wav

outputs/plots/99-45_airplane_01_selection_1_pre_before.png
outputs/plots/99-45_airplane_01_selection_1_pre_after.png
outputs/plots/99-45_airplane_01_selection_1_pre_comparison.png
outputs/plots/99-45_airplane_01_selection_1_pre_noise_profile.png
```

## How It Works

The denoising flow in `src/pipeline.py` is:

1. Load the resolved mapping CSV and look up one annotated `Selection`.
2. Load the corresponding source WAV file.
3. Crop around the annotated call using a small buffer on both sides.
4. Compute an STFT magnitude spectrogram.
5. Convert the call boundaries into spectrogram frame indices.
6. Estimate a noise profile from the frames before the call, after the call, or both.
7. Apply spectral subtraction with a floor, soft time weighting, and frequency weighting.
8. Reconstruct the waveform from the cleaned magnitude spectrogram and original phase.
9. Save the cleaned audio and diagnostic plots.

The current implementation uses:

- `N_FFT = 4096`
- `HOP_LENGTH = 512`
- `WIN_LENGTH = 4096`
- `CROP_BUFFER_SEC = 0.5`
- `ALPHA = 1.2`
- `BETA = 0.05`

These values are defined in `src/config.py`.

## Noise Modes

The pipeline supports three noise-estimation strategies:

- `pre`: estimate noise from the short window immediately before the call
- `post`: estimate noise from the short window immediately after the call
- `both`: combine the pre-call and post-call windows into one profile

This is useful because background conditions are not always symmetric. In some recordings, the pre-call region is cleaner than the post-call region, or vice versa.

## Inputs

The project expects:

- raw WAV files under `data/raw_audio/`
- annotation metadata in `data/annotations/calls.csv`
- a resolved mapping CSV at `outputs/audio_spectrogram_mapping.csv`
- reference spectrogram images under `data/spectrogram_refs/`

The mapping CSV is expected to contain:

- `Selection`
- `Sound_file`
- `Start_time`
- `End_time`
- `Call_type`
- `audio_path`
- `audio_exists`
- `expected_spectrogram`
- `spectrogram_path`
- `spectrogram_exists`

## Setup

From the `Noise Cleanup` directory:

```bash
cd "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup"
python3 -m pip install -r requirements.txt
```

Dependencies currently listed in `requirements.txt`:

- `numpy`
- `scipy`
- `librosa`
- `soundfile`
- `matplotlib`
- `pandas`
- `streamlit`

## Running The Project

### Streamlit App

The main entry point is:

```bash
cd "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup"
python3 app.py
```

If your environment is configured for Streamlit directly, this is also common:

```bash
streamlit run app.py
```

### Pipeline Module

To run the pipeline module itself, execute it from the parent directory of `src`:

```bash
cd "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup"
python3 -m src.pipeline
```

The `__main__` block in `src/pipeline.py` currently runs `Selection = 1` for all three noise modes: `pre`, `post`, and `both`.

## Important Import Note

`src/pipeline.py` uses relative imports such as:

```python
from .config import ...
```

Because of that, this command will fail:

```bash
python3 pipeline.py
```

It fails with:

```text
ImportError: attempted relative import with no known parent package
```

Why:

- running `pipeline.py` directly makes Python treat it as a standalone script
- standalone scripts do not have a package parent
- relative imports with a leading `.` only work when Python loads the file as part of a package

Use this instead:

```bash
cd "/home/jtedwards/Ubuntu_Code/Projects/Elephant_Noise/ElephantNoise/backend/Noise Cleanup"
python3 -m src.pipeline
```

If you run `python3 -m src.pipeline` from the wrong directory, Python may instead raise:

```text
ModuleNotFoundError: No module named 'src'
```

That means your current working directory is not the parent of the `src` folder.

## Interpreting Results

The most useful artifacts are usually:

- the cleaned WAV file, to hear whether broadband or steady background noise was reduced
- the comparison spectrogram, to see whether noise bands were suppressed without erasing the call
- the noise-profile plot, to understand what the subtraction step estimated as background

Good results usually look like:

- reduced stationary or slowly varying background energy
- clearer call structure in the time-frequency view
- preserved call boundaries and overall call shape

Watch for these failure modes:

- overly aggressive subtraction that thins or damages the call
- musical-noise style artifacts after subtraction
- poor noise estimates when the pre/post windows are not actually noise-only
- mismatch between annotations and the real call timing

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── annotations/
│   │   └── calls.csv
│   ├── raw_audio/
│   └── spectrogram_refs/
├── outputs/
│   ├── audio_spectrogram_mapping.csv
│   ├── cleaned_audio/
│   └── plots/
└── src/
    ├── config.py
    ├── io_utils.py
    ├── noise_estimation.py
    ├── pipeline.py
    ├── spectral_subtraction.py
    ├── spectrogram_utils.py
    ├── visualization.py
    ├── audio_spectrogram_mapping.py
    └── get_images_for_wav.py
```

## Module Summary

- `app.py`: interactive UI for browsing examples and viewing outputs
- `src/config.py`: shared paths, STFT settings, subtraction parameters, plot settings
- `src/io_utils.py`: CSV loading, selection lookup, WAV loading/saving, output naming
- `src/noise_estimation.py`: pre/post noise window and frame-range selection
- `src/spectral_subtraction.py`: noise subtraction, floors, soft masks, and weighting
- `src/spectrogram_utils.py`: STFT, magnitude/phase conversion, reconstruction helpers
- `src/visualization.py`: spectrogram and noise-profile plotting
- `src/pipeline.py`: orchestrates the full cleanup flow for one selection

## Practical Notes

- Run commands from the `Noise Cleanup` directory unless you intentionally manage `PYTHONPATH`.
- The pipeline depends on `pandas`, `librosa`, `soundfile`, and plotting libraries being installed in the active environment.
- Output directories are created automatically when the pipeline runs.
- The saved plots include call boundaries and noise-window markers to help with debugging.

## Current Outcome

At this stage, the project is set up as a reproducible denoising workflow for annotated elephant calls with both audio and visual outputs. The main measurable result is a cleaned version of each cropped call segment plus a set of plots that let you compare original and processed energy patterns.

That makes the project useful for:

- qualitative listening review
- spectrogram-based debugging
- comparing `pre`, `post`, and `both` noise estimation strategies
- building a more polished demo or research workflow on top of the current pipeline

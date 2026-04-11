# Noise Cleanup

Audio noise reduction pipeline for elephant call recordings.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Structure

```
├── app.py                    # Entry point
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw_audio/            # Input audio files
│   ├── annotations/
│   │   └── calls.csv         # Call annotation metadata
│   └── spectrogram_refs/     # Reference spectrograms
├── outputs/
│   ├── cleaned_audio/        # Processed audio output
│   └── plots/                # Generated visualizations
└── src/
    ├── config.py             # Configuration constants
    ├── io_utils.py           # File I/O utilities
    ├── spectrogram_utils.py  # Spectrogram computation helpers
    ├── noise_estimation.py   # Noise profile estimation
    ├── spectral_subtraction.py # Spectral subtraction algorithm
    ├── pipeline.py           # End-to-end processing pipeline
    └── visualization.py      # Plotting and visualization
```

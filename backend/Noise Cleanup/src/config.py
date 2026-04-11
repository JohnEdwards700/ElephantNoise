# Configuration constants for the Noise Cleanup pipeline

# Paths
RAW_AUDIO_DIR = "data/raw_audio"
ANNOTATIONS_FILE = "data/annotations/calls.csv"
SPECTROGRAM_REFS_DIR = "data/spectrogram_refs"
CLEANED_AUDIO_DIR = "outputs/cleaned_audio"
PLOTS_DIR = "outputs/plots"

# Audio processing parameters
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Noise estimation parameters
NOISE_PERCENTILE = 10  # Percentile used to estimate noise floor
OVER_SUBTRACTION_FACTOR = 1.0
SPECTRAL_FLOOR = 0.002

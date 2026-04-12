import base64
import time
from pathlib import Path
from html import escape
from typing import Dict, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src import config
from src.io_utils import load_audio, save_audio
from src.noise_estimation import estimate_noise_profile, smooth_noise_profile
from src.spectral_subtraction import spectral_subtraction
from src.spectrogram_utils import compute_stft, stft_to_magnitude_phase
from src.spectrogram_utils import reconstruct_signal


APP_DIR = Path(__file__).resolve().parent
MAPPING_CSV = APP_DIR / "outputs" / "audio_spectrogram_mapping.csv"
RAW_AUDIO_DIR = APP_DIR / "data" / "raw_audio"
SPECTROGRAM_REFS_DIR = APP_DIR / "data" / "spectrogram_refs"
DEMO_OUTPUT_DIR = APP_DIR / "outputs" / "streamlit_demo"
ORIGINAL_AUDIO_DIR = DEMO_OUTPUT_DIR / "original_audio"
CLEANED_AUDIO_DIR = DEMO_OUTPUT_DIR / "cleaned_audio"
PLOTS_DIR = DEMO_OUTPUT_DIR / "plots"

SPECTROGRAM_VIEW_OPTIONS = {
    "heatmap": "Heatmap",
    "contour": "Contour Map",
    "perspective_3d": "3D Perspective Diagram",
}

CAMERA_PRESET_OPTIONS = {
    "front_high": {"label": "Front High", "elev": 29, "azim": -92, "focal_length": 0.80},
    "front_low": {"label": "Front Low", "elev": 13, "azim": -94, "focal_length": 0.76},
    "diagonal": {"label": "Diagonal", "elev": 21, "azim": -68, "focal_length": 0.78},
    "side_low": {"label": "Side Low", "elev": 11, "azim": -19, "focal_length": 0.77},
    "side_high": {"label": "Side High", "elev": 27, "azim": -18, "focal_length": 0.80},
}
DEFAULT_CAMERA_PRESET = "diagonal"
AUTO_CAMERA_CYCLE_SECONDS = 2.4


st.set_page_config(
    page_title="Elephant Noise Cleanup",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #101813;
            --muted: #38453d;
            --forest: #203c34;
            --moss: #6d8b65;
            --sand: #efe4d2;
            --clay: #b56b45;
            --surface: rgba(255, 250, 241, 0.78);
            --surface-strong: rgba(255, 250, 241, 0.95);
            --line: rgba(32, 60, 52, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(181, 107, 69, 0.18), transparent 28%),
                radial-gradient(circle at top left, rgba(109, 139, 101, 0.20), transparent 34%),
                linear-gradient(180deg, #fbf5ea 0%, #f4ecdd 55%, #ebe2d3 100%);
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] .stMarkdown,
        [data-testid="stAppViewContainer"] .stMarkdown *,
        [data-testid="stAppViewContainer"] [data-testid="stTable"] *,
        [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
        [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] {
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] .stCaption * {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background:
                linear-gradient(180deg, rgba(32, 60, 52, 0.98), rgba(43, 73, 63, 0.96));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f6f0e6;
        }

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption {
            color: #f6f0e6 !important;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 250, 241, 0.92), rgba(250, 239, 220, 0.84));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 2rem 2rem 1.6rem 2rem;
            box-shadow: 0 24px 60px rgba(42, 53, 47, 0.10);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            display: inline-block;
            margin: 0 0 0.8rem 0;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(32, 60, 52, 0.08);
            color: var(--forest);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: 2.5rem;
            line-height: 1.05;
            color: var(--ink);
        }

        .hero p {
            margin: 0;
            max-width: 52rem;
            font-size: 1rem;
            line-height: 1.65;
            color: var(--muted);
        }

        .info-chip {
            background: rgba(255, 250, 241, 0.85);
            border: 1px solid var(--line);
            color: var(--forest);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            font-size: 0.94rem;
        }

        .info-chip strong {
            color: var(--forest);
        }

        div[data-testid="metric-container"] {
            background: var(--surface);
            border: 1px solid var(--line);
            padding: 1rem 1rem 0.8rem 1rem;
            border-radius: 22px;
            box-shadow: 0 14px 36px rgba(42, 53, 47, 0.06);
        }

        div[data-testid="metric-container"] label {
            color: var(--muted) !important;
        }

        [data-testid="stTable"] table {
            color: var(--ink);
        }

        [data-testid="stTable"] th {
            color: var(--forest);
        }

        .panel-title {
            margin: 0 0 0.6rem 0;
            color: var(--ink);
            font-size: 1.15rem;
        }

        .panel-copy {
            margin: 0 0 1rem 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .empty-panel {
            background: rgba(255, 250, 241, 0.70);
            border: 1px dashed rgba(32, 60, 52, 0.25);
            border-radius: 24px;
            padding: 1.4rem;
            color: var(--muted);
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #1f4037, #39594b);
            color: #fffaf0;
            font-weight: 700;
            min-height: 3rem;
            padding: 0.6rem 1.4rem;
            box-shadow: 0 16px 28px rgba(31, 64, 55, 0.22);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #18352d, #2f4d41);
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid var(--line);
        }

        .stAudio {
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.35rem 0.45rem;
        }

        .stExpander {
            background: rgba(255, 250, 241, 0.75);
            border: 1px solid var(--line);
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def output_stem(selection_id: int) -> str:
    return f"selection_{selection_id:03d}"


def ensure_output_dirs() -> None:
    for directory in (ORIGINAL_AUDIO_DIR, CLEANED_AUDIO_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1.0:
        signal = 0.98 * signal / peak
    return signal


def to_mono(signal: np.ndarray) -> np.ndarray:
    if np.ndim(signal) == 1:
        return np.asarray(signal, dtype=np.float32)
    return np.asarray(signal, dtype=np.float32).mean(axis=1)


def resolve_audio_path(row: pd.Series) -> Optional[Path]:
    candidates = [RAW_AUDIO_DIR / str(row["Sound_file"])]
    if pd.notna(row.get("audio_path")):
        candidates.append(Path(str(row["audio_path"])))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_reference_spectrogram(row: pd.Series) -> Optional[Path]:
    candidates = []
    if pd.notna(row.get("expected_spectrogram")):
        candidates.append(SPECTROGRAM_REFS_DIR / str(row["expected_spectrogram"]))
    if pd.notna(row.get("spectrogram_path")):
        candidates.append(Path(str(row["spectrogram_path"])))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_mapping() -> pd.DataFrame:
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"Missing mapping CSV: {MAPPING_CSV}")

    df = pd.read_csv(MAPPING_CSV).copy()
    if df.empty:
        return df

    df["Selection"] = pd.to_numeric(df["Selection"], errors="coerce").astype("Int64")
    df["Start_time"] = pd.to_numeric(df["Start_time"], errors="coerce")
    df["End_time"] = pd.to_numeric(df["End_time"], errors="coerce")
    df["duration_seconds"] = (df["End_time"] - df["Start_time"]).clip(lower=0)
    df["resolved_audio_path"] = df.apply(resolve_audio_path, axis=1)
    df["resolved_reference_spectrogram"] = df.apply(resolve_reference_spectrogram, axis=1)
    df["has_audio"] = df["resolved_audio_path"].notna()
    df["has_reference_spectrogram"] = df["resolved_reference_spectrogram"].notna()

    valid_df = df[df["has_audio"] & df["has_reference_spectrogram"]].copy()
    valid_df = valid_df.dropna(subset=["Selection", "Start_time", "End_time"])
    valid_df["Selection"] = valid_df["Selection"].astype(int)
    return valid_df.sort_values(["Call_type", "Sound_file", "Selection"]).reset_index(drop=True)


def selection_label(row: pd.Series) -> str:
    return (
        f"Selection {int(row['Selection'])} | {row['Call_type']} | "
        f"{row['Sound_file']} | {row['Start_time']:.2f}s to {row['End_time']:.2f}s"
    )


def save_spectrogram_image(
    magnitude: np.ndarray,
    sample_rate: int,
    save_path: Path,
    title: str,
) -> Path:
    db = librosa.amplitude_to_db(np.maximum(magnitude, 1e-10), ref=np.max)
    duration = magnitude.shape[1] * config.HOP_LENGTH / sample_rate

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    img = ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration, 0, sample_rate / 2],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    colorbar = fig.colorbar(img, ax=ax, pad=0.02)
    colorbar.set_label("dB")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_spectrogram_figure(
    magnitude: np.ndarray,
    sample_rate: int,
    title: str,
    view_mode: str,
    camera_preset: str = DEFAULT_CAMERA_PRESET,
) -> plt.Figure:
    db = librosa.amplitude_to_db(np.maximum(magnitude, 1e-10), ref=np.max)
    duration = magnitude.shape[1] * config.HOP_LENGTH / sample_rate
    times = np.linspace(0, duration, magnitude.shape[1])
    freqs = np.linspace(0, sample_rate / 2, magnitude.shape[0])

    if view_mode == "contour":
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        contour = ax.contourf(times, freqs, db, levels=24, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
        colorbar.set_label("dB")
        fig.tight_layout()
        return fig

    if view_mode == "perspective_3d":
        camera_settings = CAMERA_PRESET_OPTIONS.get(
            camera_preset,
            CAMERA_PRESET_OPTIONS[DEFAULT_CAMERA_PRESET],
        )
        max_time_bins = min(18, magnitude.shape[1])
        max_freq_bins = min(220, magnitude.shape[0])
        time_idx = np.linspace(0, magnitude.shape[1] - 1, max_time_bins, dtype=int)
        freq_idx = np.linspace(0, magnitude.shape[0] - 1, max_freq_bins, dtype=int)
        times_sub = times[time_idx]
        layer_positions = np.linspace(0, duration * 0.58, max_time_bins)
        freqs_sub = freqs[freq_idx]
        z_floor = float(np.percentile(db, 10) - 4.0)
        z_ceiling = float(np.percentile(db, 99.6))

        verts = []
        ridge_lines = []
        facecolors = []
        edgecolors = []

        smoothing_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smoothing_kernel /= smoothing_kernel.sum()

        for layer_index, time_bin in enumerate(time_idx):
            ridge = db[freq_idx, time_bin]
            ridge = np.convolve(ridge, smoothing_kernel, mode="same")
            ridge = z_floor + (ridge - z_floor) * 1.24
            ridge = np.clip(ridge, z_floor, z_ceiling)

            ridge_points = list(zip(freqs_sub, ridge))
            polygon = [(freqs_sub[0], z_floor), *ridge_points, (freqs_sub[-1], z_floor)]
            verts.append(polygon)
            ridge_lines.append(ridge)

            fade = layer_index / max(1, len(time_idx) - 1)
            facecolors.append((0.92, 0.94, 0.98, 0.22 + (1.0 - fade) * 0.34))
            edgecolors.append((1.0, 1.0, 1.0, 0.22 + (1.0 - fade) * 0.30))

        fig = plt.figure(figsize=(9.2, 5.9))
        fig.patch.set_facecolor("#101317")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#101317")

        collection = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=1.0,
        )
        ax.add_collection3d(collection, zs=layer_positions, zdir="y")

        for layer_y, ridge, fade in zip(layer_positions, ridge_lines, np.linspace(1.0, 0.35, len(ridge_lines))):
            ax.plot(
                freqs_sub,
                np.full_like(freqs_sub, layer_y),
                ridge,
                color=(1.0, 1.0, 1.0, 0.16 + 0.42 * fade),
                linewidth=1.35,
            )

        x_min = float(freqs_sub.min())
        x_max = float(freqs_sub.max())
        y_min = float(layer_positions.min())
        y_max = float(layer_positions.max())
        x_span = x_max - x_min
        y_span = max(y_max - y_min, 1e-6)

        ax.set_xlim(x_min - 0.015 * x_span, x_max + 0.01 * x_span)
        ax.set_ylim(y_min - 0.03 * y_span, y_max + 0.02 * y_span)
        ax.set_zlim(z_floor, z_ceiling)
        ax.set_title(title, color="#f4f6f8", pad=18, fontsize=13)
        ax.set_xlabel("Frequency (Hz)", labelpad=10, color="#d7dde3")
        ax.set_ylabel("Time (s)", labelpad=10, color="#d7dde3")
        ax.set_zlabel("Intensity (dB)", labelpad=8, color="#d7dde3")
        ax.view_init(
            elev=camera_settings["elev"],
            azim=camera_settings["azim"],
        )
        try:
            ax.set_proj_type("persp", focal_length=camera_settings["focal_length"])
        except TypeError:
            ax.set_proj_type("persp")
        ax.set_box_aspect((2.35, 0.92, 0.78))

        ax.xaxis._axinfo["grid"]["color"] = (0.42, 0.46, 0.50, 0.42)
        ax.yaxis._axinfo["grid"]["color"] = (0.42, 0.46, 0.50, 0.35)
        ax.zaxis._axinfo["grid"]["color"] = (0.30, 0.34, 0.38, 0.16)
        ax.xaxis._axinfo["tick"]["color"] = (0.84, 0.88, 0.91, 0.95)
        ax.yaxis._axinfo["tick"]["color"] = (0.84, 0.88, 0.91, 0.95)
        ax.zaxis._axinfo["tick"]["color"] = (0.84, 0.88, 0.91, 0.95)

        ax.xaxis.set_pane_color((0.08, 0.10, 0.12, 0.78))
        ax.yaxis.set_pane_color((0.08, 0.10, 0.12, 0.24))
        ax.zaxis.set_pane_color((0.08, 0.10, 0.12, 0.0))

        x_ticks = np.linspace(freqs_sub.min(), freqs_sub.max(), 8)
        y_ticks = np.linspace(layer_positions.min(), layer_positions.max(), 4)
        z_ticks = np.linspace(z_floor, z_ceiling, 4)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)
        ax.set_xticklabels([f"{tick:.0f}" for tick in x_ticks], rotation=0, ha="center")
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(times_sub.min(), times_sub.max(), 4)])
        ax.tick_params(colors="#d7dde3", labelsize=9, pad=2)

        ax.xaxis.line.set_color((0.72, 0.76, 0.80, 0.62))
        ax.yaxis.line.set_color((0.72, 0.76, 0.80, 0.50))
        ax.zaxis.line.set_color((0.50, 0.58, 0.64, 0.18))

        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    image = ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration, 0, sample_rate / 2],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("dB")
    fig.tight_layout()
    return fig


def load_clip_from_selection(row: pd.Series) -> Tuple[np.ndarray, int]:
    audio_path = Path(row["resolved_audio_path"])
    signal, sample_rate = load_audio(str(audio_path), sample_rate=config.SAMPLE_RATE)
    signal = to_mono(signal)

    start_sample = max(0, int(float(row["Start_time"]) * sample_rate))
    end_sample = min(len(signal), int(float(row["End_time"]) * sample_rate))
    if end_sample <= start_sample:
        raise ValueError(
            f"Selection {int(row['Selection'])} produced an empty clip. "
            "Check the mapping timestamps."
        )

    clip = signal[start_sample:end_sample]
    if clip.size == 0:
        raise ValueError(
            f"Selection {int(row['Selection'])} produced an empty clip. "
            "Check the mapping timestamps."
        )

    return normalize_audio(clip), sample_rate


def prepare_original_assets(row: pd.Series) -> Dict[str, object]:
    ensure_output_dirs()
    selection_id = int(row["Selection"])
    stem = output_stem(selection_id)
    original_audio_path = ORIGINAL_AUDIO_DIR / f"{stem}_original.wav"
    original_spectrogram_path = PLOTS_DIR / f"{stem}_original.png"

    clip, sample_rate = load_clip_from_selection(row)
    original_stft = compute_stft(
        clip,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    original_magnitude, _ = stft_to_magnitude_phase(original_stft)

    if not original_audio_path.exists():
        save_audio(str(original_audio_path), clip, sample_rate)
    if not original_spectrogram_path.exists():
        save_spectrogram_image(
            magnitude=original_magnitude,
            sample_rate=sample_rate,
            save_path=original_spectrogram_path,
            title=f"Original spectrogram | Selection {selection_id}",
        )

    return {
        "original_audio_path": original_audio_path,
        "original_spectrogram_path": original_spectrogram_path,
        "original_magnitude": original_magnitude,
        "sample_rate": sample_rate,
    }


def process_call(row: pd.Series) -> dict[str, object]:
    ensure_output_dirs()
    selection_id = int(row["Selection"])
    stem = output_stem(selection_id)

    preview_assets = prepare_original_assets(row)
    clip, sample_rate = load_clip_from_selection(row)

    stft_matrix = compute_stft(
        clip,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    magnitude, phase = stft_to_magnitude_phase(stft_matrix)

    noise_profile = estimate_noise_profile(
        magnitude,
        percentile=config.NOISE_PERCENTILE,
    )
    noise_profile = smooth_noise_profile(noise_profile)
    cleaned_magnitude = spectral_subtraction(
        magnitude,
        noise_profile,
        over_subtraction=config.OVER_SUBTRACTION_FACTOR,
        spectral_floor=config.SPECTRAL_FLOOR,
    )

    cleaned_signal = reconstruct_signal(
        cleaned_magnitude,
        phase,
        hop_length=config.HOP_LENGTH,
    )
    cleaned_signal = normalize_audio(cleaned_signal)
    if len(cleaned_signal) < len(clip):
        cleaned_signal = np.pad(cleaned_signal, (0, len(clip) - len(cleaned_signal)))
    cleaned_signal = cleaned_signal[: len(clip)]

    cleaned_audio_path = CLEANED_AUDIO_DIR / f"{stem}_cleaned.wav"
    cleaned_spectrogram_path = PLOTS_DIR / f"{stem}_cleaned.png"

    save_audio(str(cleaned_audio_path), cleaned_signal, sample_rate)
    save_spectrogram_image(
        magnitude=cleaned_magnitude,
        sample_rate=sample_rate,
        save_path=cleaned_spectrogram_path,
        title=f"Cleaned spectrogram | Selection {selection_id}",
    )

    return {
        "selection_id": selection_id,
        "sample_rate": sample_rate,
        "original_audio_path": preview_assets["original_audio_path"],
        "original_spectrogram_path": preview_assets["original_spectrogram_path"],
        "original_magnitude": preview_assets["original_magnitude"],
        "cleaned_audio_path": cleaned_audio_path,
        "cleaned_spectrogram_path": cleaned_spectrogram_path,
        "cleaned_magnitude": cleaned_magnitude,
        "reference_spectrogram_path": Path(row["resolved_reference_spectrogram"]),
    }


def audio_player(path: Path, label: str) -> None:
    audio_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
    safe_label = escape(label)
    component_id = "".join(
        char if char.isalnum() else "-"
        for char in f"{path.stem}-{label}".lower()
    )

    components.html(
        f"""
        <div class="audio-card">
            <div class="audio-card__header">
                <div class="audio-card__label">{safe_label}</div>
                <div class="audio-card__hint">Press play to activate the live audio visualizer.</div>
            </div>
            <canvas id="viz-{component_id}" width="720" height="120"></canvas>
            <audio id="audio-{component_id}" controls preload="metadata">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav" />
            </audio>
        </div>
        <script>
        const audio = document.getElementById("audio-{component_id}");
        const canvas = document.getElementById("viz-{component_id}");
        const context = canvas.getContext("2d");
        const width = canvas.width;
        const height = canvas.height;
        let audioContext;
        let analyser;
        let source;
        let dataArray;
        let animationFrame;

        function drawFrame(values, active) {{
            const background = context.createLinearGradient(0, 0, width, height);
            background.addColorStop(0, active ? "#17352c" : "#edf1ea");
            background.addColorStop(1, active ? "#315747" : "#dce4d7");
            context.fillStyle = background;
            context.fillRect(0, 0, width, height);

            const barWidth = width / values.length;
            for (let i = 0; i < values.length; i += 1) {{
                const intensity = values[i] / 255;
                const barHeight = Math.max(8, intensity * (height - 18));
                const x = i * barWidth;
                const y = height - barHeight;
                context.fillStyle = active ? "#f5e6b3" : "#7d8f7d";
                context.fillRect(x + 1.5, y, Math.max(2, barWidth - 3), barHeight);
            }}
        }}

        function drawIdle() {{
            drawFrame(new Array(48).fill(42), false);
        }}

        async function ensureAudioGraph() {{
            if (!audioContext) {{
                const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                audioContext = new AudioContextClass();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 128;
                analyser.smoothingTimeConstant = 0.82;
                source = audioContext.createMediaElementSource(audio);
                source.connect(analyser);
                analyser.connect(audioContext.destination);
                dataArray = new Uint8Array(analyser.frequencyBinCount);
            }}

            if (audioContext.state === "suspended") {{
                await audioContext.resume();
            }}
        }}

        function animate() {{
            if (!analyser || audio.paused || audio.ended) {{
                drawIdle();
                return;
            }}

            analyser.getByteFrequencyData(dataArray);
            drawFrame(Array.from(dataArray), true);
            animationFrame = window.requestAnimationFrame(animate);
        }}

        async function startVisualizer() {{
            try {{
                await ensureAudioGraph();
                window.cancelAnimationFrame(animationFrame);
                animate();
            }} catch (error) {{
                drawIdle();
                console.error("Visualizer failed to start", error);
            }}
        }}

        function stopVisualizer() {{
            window.cancelAnimationFrame(animationFrame);
            drawIdle();
        }}

        audio.addEventListener("play", startVisualizer);
        audio.addEventListener("pause", stopVisualizer);
        audio.addEventListener("ended", stopVisualizer);

        drawIdle();
        </script>
        <style>
        body {{
            margin: 0;
            font-family: "Segoe UI", sans-serif;
            background: transparent;
        }}

        .audio-card {{
            background: rgba(255, 250, 241, 0.96);
            border: 1px solid rgba(32, 60, 52, 0.14);
            border-radius: 18px;
            padding: 0.95rem 1rem 1rem 1rem;
            box-shadow: 0 14px 34px rgba(42, 53, 47, 0.08);
        }}

        .audio-card__header {{
            margin-bottom: 0.6rem;
        }}

        .audio-card__label {{
            color: #122019;
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}

        .audio-card__hint {{
            color: #415045;
            font-size: 0.83rem;
            line-height: 1.4;
        }}

        .audio-card * {{
            box-sizing: border-box;
        }}

        canvas {{
            width: 100%;
            height: 120px;
            display: block;
            border-radius: 14px;
            margin: 0.75rem 0 0.9rem 0;
            border: 1px solid rgba(18, 32, 25, 0.10);
        }}

        audio {{
            width: 100%;
            min-height: 54px;
            display: block;
        }}
        </style>
        """,
        height=280,
    )


def render_panel(
    title: str,
    copy: str,
    spectrogram_path: Optional[Path] = None,
    spectrogram_magnitude: Optional[np.ndarray] = None,
    spectrogram_sample_rate: Optional[int] = None,
    spectrogram_view_mode: str = "heatmap",
    camera_preset: str = DEFAULT_CAMERA_PRESET,
    audio_path: Optional[Path] = None,
    image_caption: Optional[str] = None,
    audio_caption: Optional[str] = None,
) -> None:
    st.markdown(f"<h3 class='panel-title'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p class='panel-copy'>{copy}</p>", unsafe_allow_html=True)

    if spectrogram_magnitude is not None and spectrogram_sample_rate is not None:
        figure = build_spectrogram_figure(
            magnitude=spectrogram_magnitude,
            sample_rate=spectrogram_sample_rate,
            title=f"{title} | {SPECTROGRAM_VIEW_OPTIONS.get(spectrogram_view_mode, 'Heatmap')}",
            view_mode=spectrogram_view_mode,
            camera_preset=camera_preset,
        )
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        if image_caption:
            st.caption(image_caption)
    elif spectrogram_path and spectrogram_path.exists():
        st.image(str(spectrogram_path), use_container_width=True)
        if image_caption:
            st.caption(image_caption)
    else:
        st.markdown(
            "<div class='empty-panel'>No spectrogram is available yet.</div>",
            unsafe_allow_html=True,
        )

    if audio_path and audio_path.exists():
        audio_player(audio_path, audio_caption or "Audio")
    else:
        st.markdown(
            "<div class='empty-panel'>Run cleaning to generate the audio preview.</div>",
            unsafe_allow_html=True,
        )


def render_selected_call_metadata(row: pd.Series) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selection", f"#{int(row['Selection'])}")
    metric_columns[1].metric("Call Type", str(row["Call_type"]).title())
    metric_columns[2].metric("Clip Window", f"{row['Start_time']:.2f}s to {row['End_time']:.2f}s")
    metric_columns[3].metric("Duration", f"{row['duration_seconds']:.2f}s")

    st.markdown(
        f"""
        <div class="info-chip">
            Selected file: <strong>{row['Sound_file']}</strong><br>
            Demo outputs are written to <strong>{DEMO_OUTPUT_DIR}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Selection details", expanded=False):
        details = pd.DataFrame(
            [
                ("Selection", int(row["Selection"])),
                ("Call type", row["Call_type"]),
                ("Sound file", row["Sound_file"]),
                ("Start time (s)", f"{row['Start_time']:.4f}"),
                ("End time (s)", f"{row['End_time']:.4f}"),
                ("Duration (s)", f"{row['duration_seconds']:.4f}"),
                ("Audio source", str(row["resolved_audio_path"])),
                ("Reference spectrogram", str(row["resolved_reference_spectrogram"])),
            ],
            columns=["Field", "Value"],
        )
        st.table(details)


def render_sidebar(df: pd.DataFrame) -> pd.Series:
    st.sidebar.markdown("## Explore Calls")
    st.sidebar.caption(
        "Filter by call type and source file, then choose a valid labeled example."
    )

    call_type_options = ["All call types"] + sorted(df["Call_type"].dropna().unique())
    selected_call_type = st.sidebar.selectbox("Call type filter", call_type_options)

    filtered_df = df.copy()
    if selected_call_type != "All call types":
        filtered_df = filtered_df[filtered_df["Call_type"] == selected_call_type]

    sound_file_options = ["All sound files"] + sorted(filtered_df["Sound_file"].unique())
    selected_sound_file = st.sidebar.selectbox("Sound file filter", sound_file_options)

    if selected_sound_file != "All sound files":
        filtered_df = filtered_df[filtered_df["Sound_file"] == selected_sound_file]

    filtered_df = filtered_df.sort_values(["Sound_file", "Selection"]).reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError("No valid selections match the current filters.")

    selection_lookup = {
        int(row["Selection"]): selection_label(row)
        for _, row in filtered_df.iterrows()
    }
    selected_selection = st.sidebar.selectbox(
        "Selection dropdown",
        options=list(selection_lookup.keys()),
        format_func=lambda selection_id: selection_lookup[selection_id],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(filtered_df)} selections across {filtered_df['Sound_file'].nunique()} files."
    )

    return filtered_df[filtered_df["Selection"] == selected_selection].iloc[0]


def main() -> None:
    inject_styles()
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Streamlit Demo</div>
            <h1>Elephant Noise Cleanup Studio</h1>
            <p>
                Choose a verified elephant call from the sidebar, review its metadata,
                and run a focused cleanup pass on that exact time window. The app keeps
                the experience simple: original versus cleaned spectrograms, original
                versus cleaned audio, and the key details needed to judge the result.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        mapping_df = load_mapping()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if mapping_df.empty:
        st.warning("The mapping CSV loaded successfully, but it does not contain valid selections.")
        return

    try:
        selected_row = render_sidebar(mapping_df)
    except ValueError as exc:
        st.warning(str(exc))
        return

    current_selection = int(selected_row["Selection"])
    if st.session_state.get("active_selection") != current_selection:
        st.session_state["active_selection"] = current_selection
        st.session_state.pop("processing_result", None)

    render_selected_call_metadata(selected_row)

    preview_assets = prepare_original_assets(selected_row)

    spectrogram_view_mode = st.radio(
        "Spectrogram view",
        options=list(SPECTROGRAM_VIEW_OPTIONS.keys()),
        format_func=lambda option: SPECTROGRAM_VIEW_OPTIONS[option],
        horizontal=True,
    )
    st.caption(
        "Switch between the standard heatmap, a contour-style view, and a 3D perspective diagram."
    )

    st.session_state.setdefault("camera_preset_selector", DEFAULT_CAMERA_PRESET)
    st.session_state.setdefault("auto_cycle_cameras", False)
    st.session_state.setdefault("last_camera_cycle_at", 0.0)

    action_columns = st.columns([0.9, 2.1])
    with action_columns[0]:
        run_requested = st.button("Run Cleaning", type="primary", use_container_width=True)
    with action_columns[1]:
        st.markdown(
            """
            <div class="info-chip">
                Cleaning runs only on the selected call window and saves reusable demo
                artifacts so you can revisit the same example without hunting for files.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_requested:
        with st.spinner("Cleaning the selected call..."):
            try:
                st.session_state["processing_result"] = process_call(selected_row)
            except Exception as exc:
                st.error(f"Cleaning failed: {exc}")

    result = st.session_state.get("processing_result")

    @st.fragment(run_every=2.4)
    def render_spectrogram_workspace() -> None:
        preset_keys = list(CAMERA_PRESET_OPTIONS.keys())

        if st.session_state.get("camera_preset_selector") not in preset_keys:
            st.session_state["camera_preset_selector"] = DEFAULT_CAMERA_PRESET

        auto_cycle_enabled = bool(st.session_state.get("auto_cycle_cameras", False))
        auto_cycle_active = auto_cycle_enabled and spectrogram_view_mode == "perspective_3d"

        if auto_cycle_active:
            now = time.time()
            last_cycle_at = float(st.session_state.get("last_camera_cycle_at", 0.0))
            if last_cycle_at <= 0.0:
                st.session_state["last_camera_cycle_at"] = now
            elif now - last_cycle_at >= AUTO_CAMERA_CYCLE_SECONDS:
                current_preset = st.session_state.get("camera_preset_selector", DEFAULT_CAMERA_PRESET)
                current_index = preset_keys.index(current_preset)
                next_preset = preset_keys[(current_index + 1) % len(preset_keys)]
                st.session_state["camera_preset_selector"] = next_preset
                st.session_state["last_camera_cycle_at"] = now
        else:
            st.session_state["last_camera_cycle_at"] = 0.0

        if spectrogram_view_mode == "perspective_3d":
            camera_columns = st.columns([1.8, 0.9], gap="large")
            with camera_columns[0]:
                selected_camera_preset = st.radio(
                    "3D camera preset",
                    options=preset_keys,
                    format_func=lambda preset: CAMERA_PRESET_OPTIONS[preset]["label"],
                    horizontal=True,
                    key="camera_preset_selector",
                )
            with camera_columns[1]:
                st.toggle(
                    "Auto cycle perspectives",
                    key="auto_cycle_cameras",
                    help="Cycle through the 3D camera presets automatically. Turn this off any time.",
                )

            if st.session_state.get("auto_cycle_cameras", False):
                st.caption("Camera auto-cycle is on. Disable the toggle to stop rotating through presets.")
        else:
            selected_camera_preset = st.session_state.get("camera_preset_selector", DEFAULT_CAMERA_PRESET)
            if st.session_state.get("auto_cycle_cameras", False):
                st.caption("Auto-cycle is available in the 3D spectrogram view.")

        result_columns = st.columns(2, gap="large")
        with result_columns[0]:
            render_panel(
                title="Original Call",
                copy=(
                    "Baseline view of the selected elephant call segment before noise "
                    "reduction. This uses the clip extracted from the mapping timestamps."
                ),
                spectrogram_path=preview_assets["original_spectrogram_path"],
                spectrogram_magnitude=preview_assets["original_magnitude"],
                spectrogram_sample_rate=preview_assets["sample_rate"],
                spectrogram_view_mode=spectrogram_view_mode,
                camera_preset=selected_camera_preset,
                audio_path=preview_assets["original_audio_path"],
                image_caption="Original spectrogram",
                audio_caption="Original audio",
            )

        with result_columns[1]:
            if result and result["selection_id"] == current_selection:
                render_panel(
                    title="Cleaned Call",
                    copy=(
                        "Noise-reduced output generated from the selected clip using the "
                        "current spectral subtraction settings."
                    ),
                    spectrogram_path=result["cleaned_spectrogram_path"],
                    spectrogram_magnitude=result["cleaned_magnitude"],
                    spectrogram_sample_rate=result["sample_rate"],
                    spectrogram_view_mode=spectrogram_view_mode,
                    camera_preset=selected_camera_preset,
                    audio_path=result["cleaned_audio_path"],
                    image_caption="Cleaned spectrogram",
                    audio_caption="Cleaned audio",
                )
                st.caption(f"Reference spectrogram on file: {result['reference_spectrogram_path']}")
            else:
                render_panel(
                    title="Cleaned Call",
                    copy=(
                        "Run cleaning to generate the denoised clip and its spectrogram for "
                        "this selection."
                    ),
                )

    render_spectrogram_workspace()


if __name__ == "__main__":
    main()

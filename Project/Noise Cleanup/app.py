import base64
import io
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src import config
from src.io_utils import get_call_by_selection, load_audio, load_mapping_csv
from src.spectrogram_utils import crop_audio


APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
CLEANED_AUDIO_DIR = OUTPUTS_DIR / "cleaned_audio"
PLOTS_DIR = OUTPUTS_DIR / "plots"

NOISE_MODE_ORDER = ("pre", "post", "both")
NOISE_MODE_LABELS = {
    "pre": "Pre-Noise Cleanup",
    "post": "Post-Noise Cleanup",
    "both": "Combined Cleanup",
}
NOISE_MODE_COPY = {
    "pre": "Noise is estimated only from the audio just before the annotated call.",
    "post": "Noise is estimated only from the audio just after the annotated call.",
    "both": "Noise is estimated from both sides of the call and blended into one profile.",
}
PREFERRED_SELECTION = 1
WAVE_PERSPECTIVES = {
    "isometric": {"label": "Isometric", "elev": 30, "azim": -61},
    "front_arc": {"label": "Front Arc", "elev": 18, "azim": -95},
    "side_scan": {"label": "Side Scan", "elev": 13, "azim": -18},
    "top_map": {"label": "Top Map", "elev": 84, "azim": -90},
}

AUDIO_FILE_PATTERN = re.compile(
    r"^(?P<sound_stem>.+)_selection_(?P<selection>\d+)_(?P<mode>pre|post|both)_cleaned\.wav$"
)
PLOT_FILE_PATTERN = re.compile(
    r"^(?P<sound_stem>.+)_selection_(?P<selection>\d+)_(?P<mode>pre|post|both)_(?P<plot_kind>before|after|comparison|noise_profile)\.png$"
)

st.set_page_config(
    page_title="Elephant Noise Cleanup",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass(frozen=True)
class ModeArtifacts:
    mode: str
    audio_path: Path
    before_plot_path: Path
    after_plot_path: Path
    comparison_plot_path: Path
    noise_profile_plot_path: Path


@dataclass(frozen=True)
class DemoExample:
    selection: int
    sound_file: str
    sound_stem: str
    call_type: str
    start_time: float
    end_time: float
    crop_start: float
    crop_end: float
    source_audio_path: Path
    reference_spectrogram_path: Optional[Path]
    original_before_plot_path: Path
    original_audio_bytes: bytes
    modes: Dict[str, ModeArtifacts]
    available_group_count: int


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {
            --bg: #06131f;
            --bg-2: #0b1c2b;
            --bg-3: #11283d;
            --ink: #e8f6ff;
            --muted: #8ba6bb;
            --accent: #6df7ff;
            --accent-2: #5c8cff;
            --accent-3: #9a7bff;
            --surface: rgba(10, 22, 36, 0.76);
            --surface-strong: rgba(15, 30, 48, 0.92);
            --line: rgba(109, 247, 255, 0.16);
            --glow: 0 0 0 1px rgba(109, 247, 255, 0.08), 0 18px 48px rgba(0, 0, 0, 0.28);
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 18%, rgba(109, 247, 255, 0.18), transparent 24%),
                radial-gradient(circle at 82% 12%, rgba(154, 123, 255, 0.20), transparent 26%),
                radial-gradient(circle at 76% 78%, rgba(92, 140, 255, 0.18), transparent 24%),
                linear-gradient(160deg, var(--bg) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] .stMarkdown,
        [data-testid="stAppViewContainer"] .stMarkdown *,
        [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
        [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] {
            color: var(--ink) !important;
            font-family: "Space Grotesk", sans-serif;
        }

        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] .stCaption * {
            color: var(--muted) !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top left, rgba(109, 247, 255, 0.18), transparent 28%),
                radial-gradient(circle at bottom right, rgba(154, 123, 255, 0.14), transparent 32%),
                linear-gradient(145deg, rgba(13, 27, 43, 0.94), rgba(9, 18, 31, 0.88));
            border: 1px solid var(--line);
            border-radius: 30px;
            padding: 2.2rem 2.2rem 1.8rem 2.2rem;
            box-shadow: var(--glow);
            margin-bottom: 1.4rem;
            backdrop-filter: blur(18px);
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0%, rgba(109, 247, 255, 0.08) 48%, transparent 100%);
            transform: translateX(-100%);
            animation: hero-scan 8s linear infinite;
            pointer-events: none;
        }

        @keyframes hero-scan {
            to {
                transform: translateX(100%);
            }
        }

        .hero-kicker {
            display: inline-block;
            margin: 0 0 0.8rem 0;
            padding: 0.34rem 0.78rem;
            border-radius: 999px;
            background: rgba(109, 247, 255, 0.10);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            border: 1px solid rgba(109, 247, 255, 0.18);
            box-shadow: inset 0 0 22px rgba(109, 247, 255, 0.06);
        }

        .hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: clamp(2.5rem, 4vw, 4.3rem);
            line-height: 1.05;
            color: var(--ink);
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0;
            max-width: 54rem;
            font-size: 1.02rem;
            line-height: 1.75;
            color: var(--muted);
        }

        .info-chip {
            background: linear-gradient(145deg, rgba(12, 27, 42, 0.88), rgba(8, 18, 31, 0.84));
            border: 1px solid var(--line);
            color: var(--ink);
            border-radius: 18px;
            padding: 1rem 1.08rem;
            margin-bottom: 1.1rem;
            font-size: 0.94rem;
            line-height: 1.6;
            box-shadow: var(--glow);
            backdrop-filter: blur(18px);
        }

        .info-chip strong {
            color: var(--accent);
            font-weight: 700;
        }

        .section-title {
            margin: 1.4rem 0 0.4rem 0;
            color: var(--ink);
            font-size: 1.42rem;
            letter-spacing: -0.03em;
        }

        .section-copy {
            margin: 0 0 1rem 0;
            color: var(--muted);
            font-size: 0.97rem;
            line-height: 1.72;
        }

        .mode-pill {
            display: inline-block;
            padding: 0.3rem 0.78rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(109, 247, 255, 0.16), rgba(92, 140, 255, 0.14));
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(109, 247, 255, 0.18);
        }

        div[data-testid="metric-container"] {
            background: linear-gradient(160deg, rgba(14, 28, 45, 0.86), rgba(8, 19, 31, 0.84));
            border: 1px solid var(--line);
            padding: 1rem 1rem 0.8rem 1rem;
            border-radius: 22px;
            box-shadow: var(--glow);
            backdrop-filter: blur(16px);
        }

        div[data-testid="metric-container"] label {
            color: var(--muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.7rem !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-size: 1.28rem !important;
        }

        [data-testid="stImage"] img {
            border-radius: 20px;
            border: 1px solid var(--line);
            box-shadow: var(--glow);
        }

        .empty-panel {
            background: rgba(9, 20, 33, 0.72);
            border: 1px dashed rgba(109, 247, 255, 0.24);
            border-radius: 24px;
            padding: 1.4rem;
            color: var(--muted);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(14, 28, 45, 0.72);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.55rem 0.95rem;
            color: var(--muted);
            transition: all 180ms ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            border-color: rgba(109, 247, 255, 0.34);
            color: var(--ink);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(109, 247, 255, 0.18), rgba(92, 140, 255, 0.18)) !important;
            color: var(--ink) !important;
            box-shadow: inset 0 0 18px rgba(109, 247, 255, 0.08);
        }

        .stDivider {
            opacity: 0.32;
        }

        .stExpander {
            background: rgba(12, 26, 41, 0.74);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--glow);
        }

        .stTable table {
            background: transparent;
        }

        .stTable th {
            color: var(--accent) !important;
            background: rgba(109, 247, 255, 0.06);
        }

        .stTable td {
            color: var(--ink) !important;
            border-color: rgba(109, 247, 255, 0.08) !important;
        }

        code {
            color: var(--accent);
            font-family: "JetBrains Mono", monospace;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 1.6rem 1.25rem 1.35rem 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_audio_bytes(signal, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def to_mono(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 1:
        return array
    return array.mean(axis=1)


@st.cache_data(show_spinner=False)
def load_waveform_signal(audio_path: str) -> Tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(audio_path)
    signal = to_mono(signal)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32), int(sample_rate)


def build_wave_layers(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_points = 2200
    if signal.size > max_points:
        sample_index = np.linspace(0, signal.size - 1, max_points, dtype=int)
        signal = signal[sample_index]

    times = np.linspace(0, len(signal) / sample_rate, len(signal), dtype=np.float32)

    segments = 16
    edges = np.linspace(0, len(signal), segments + 1, dtype=int)
    smoothing_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    smoothing_kernel /= smoothing_kernel.sum()

    layer_positions = np.linspace(0.0, max(times[-1] * 0.68, 0.8), segments, dtype=np.float32)
    layered_wave = []
    for index in range(segments):
        start = edges[index]
        end = max(start + 1, edges[index + 1])
        segment = signal[start:end]
        smoothed = np.convolve(segment, smoothing_kernel, mode="same")
        interp_positions = np.linspace(0, len(smoothed) - 1, len(times))
        source_positions = np.arange(len(smoothed))
        layer = np.interp(interp_positions, source_positions, smoothed)
        fade = 1.0 - (index / max(1, segments - 1)) * 0.72
        layered_wave.append(layer * fade)

    return times, layer_positions, np.asarray(layered_wave, dtype=np.float32)


def build_wave_model_figure(audio_path: str, perspective_key: str) -> plt.Figure:
    signal, sample_rate = load_waveform_signal(audio_path)
    times, layer_positions, layered_wave = build_wave_layers(signal, sample_rate)

    z_floor = -0.25
    verts = []
    facecolors = []
    ridge_colors = []

    for index, layer in enumerate(layered_wave):
        ridge_points = list(zip(times, layer))
        polygon = [(times[0], z_floor), *ridge_points, (times[-1], z_floor)]
        verts.append(polygon)

        glow_strength = 1.0 - (index / max(1, len(layered_wave) - 1)) * 0.65
        facecolors.append((0.08, 0.92, 0.98, 0.08 + 0.18 * glow_strength))
        ridge_colors.append((0.49, 0.97, 1.0, 0.18 + 0.50 * glow_strength))

    figure = plt.figure(figsize=(10.6, 5.6))
    figure.patch.set_facecolor("#07111c")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#07111c")

    collection = PolyCollection(
        verts,
        facecolors=facecolors,
        edgecolors=(0.43, 0.97, 1.0, 0.04),
        linewidths=0.6,
    )
    axis.add_collection3d(collection, zs=layer_positions, zdir="y")

    for y_position, layer, ridge_color in zip(layer_positions, layered_wave, ridge_colors):
        axis.plot(
            times,
            np.full_like(times, y_position),
            layer,
            color=ridge_color,
            linewidth=1.2,
        )

    base_projection = np.percentile(np.abs(layered_wave), 78, axis=0)
    axis.plot(
        times,
        np.full_like(times, layer_positions[-1] + 0.06 * max(layer_positions[-1], 1.0)),
        base_projection * 0.10 - 0.22,
        color=(0.36, 0.56, 1.0, 0.30),
        linewidth=1.1,
    )

    axis.set_xlim(float(times.min()), float(times.max()))
    axis.set_ylim(float(layer_positions.min()) - 0.06, float(layer_positions.max()) + 0.09)
    axis.set_zlim(z_floor, 1.05)
    axis.set_box_aspect((2.9, 1.0, 0.9))

    settings = WAVE_PERSPECTIVES[perspective_key]
    axis.view_init(elev=settings["elev"], azim=settings["azim"])

    try:
        axis.set_proj_type("persp", focal_length=0.88)
    except TypeError:
        axis.set_proj_type("persp")

    axis.set_title("3D waveform model", color="#eaf8ff", pad=14, fontsize=13)
    axis.set_xlabel("Time (s)", color="#93bdd0", labelpad=10)
    axis.set_ylabel("Depth", color="#93bdd0", labelpad=10)
    axis.set_zlabel("Amplitude", color="#93bdd0", labelpad=8)

    axis.xaxis._axinfo["grid"]["color"] = (0.41, 0.83, 0.98, 0.14)
    axis.yaxis._axinfo["grid"]["color"] = (0.41, 0.83, 0.98, 0.10)
    axis.zaxis._axinfo["grid"]["color"] = (0.41, 0.83, 0.98, 0.08)

    axis.xaxis._axinfo["tick"]["color"] = (0.85, 0.96, 1.0, 0.85)
    axis.yaxis._axinfo["tick"]["color"] = (0.85, 0.96, 1.0, 0.55)
    axis.zaxis._axinfo["tick"]["color"] = (0.85, 0.96, 1.0, 0.78)

    axis.xaxis.set_pane_color((0.04, 0.08, 0.13, 0.90))
    axis.yaxis.set_pane_color((0.05, 0.11, 0.18, 0.22))
    axis.zaxis.set_pane_color((0.05, 0.11, 0.18, 0.04))

    axis.tick_params(colors="#dff6ff", labelsize=8, pad=2)
    axis.xaxis.line.set_color((0.41, 0.83, 0.98, 0.25))
    axis.yaxis.line.set_color((0.41, 0.83, 0.98, 0.15))
    axis.zaxis.line.set_color((0.41, 0.83, 0.98, 0.20))

    figure.tight_layout()
    return figure


def audio_player(audio_bytes: bytes, label: str, hint: str) -> None:
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    safe_label = escape(label)
    safe_hint = escape(hint)
    component_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "audio"

    components.html(
        f"""
        <div class="audio-card">
            <div class="audio-card__header">
                <div class="audio-card__label">{safe_label}</div>
                <div class="audio-card__hint">{safe_hint}</div>
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
            background.addColorStop(0, active ? "#081624" : "#0b1827");
            background.addColorStop(1, active ? "#112c43" : "#132336");
            context.fillStyle = background;
            context.fillRect(0, 0, width, height);

            const barWidth = width / values.length;
            for (let i = 0; i < values.length; i += 1) {{
                const intensity = values[i] / 255;
                const barHeight = Math.max(8, intensity * (height - 18));
                const x = i * barWidth;
                const y = height - barHeight;
                context.fillStyle = active ? "#72f7ff" : "#4d6780";
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
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            background: transparent;
        }}

        .audio-card {{
            background: linear-gradient(145deg, rgba(12, 25, 40, 0.94), rgba(9, 18, 31, 0.92));
            border: 1px solid rgba(109, 247, 255, 0.16);
            border-radius: 20px;
            padding: 0.95rem 1rem 1rem 1rem;
            box-shadow: 0 0 0 1px rgba(109, 247, 255, 0.05), 0 18px 44px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(16px);
        }}

        .audio-card__header {{
            margin-bottom: 0.6rem;
        }}

        .audio-card__label {{
            color: #ecf8ff;
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }}

        .audio-card__hint {{
            color: #90abc0;
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
            border-radius: 16px;
            margin: 0.75rem 0 0.9rem 0;
            border: 1px solid rgba(109, 247, 255, 0.12);
            box-shadow: inset 0 0 20px rgba(109, 247, 255, 0.04);
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


def truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def discover_output_groups() -> Dict[Tuple[str, int], dict]:
    groups: Dict[Tuple[str, int], dict] = {}

    for audio_path in sorted(CLEANED_AUDIO_DIR.glob("*_cleaned.wav")):
        match = AUDIO_FILE_PATTERN.match(audio_path.name)
        if not match:
            continue

        sound_stem = match.group("sound_stem")
        selection = int(match.group("selection"))
        mode = match.group("mode")
        key = (sound_stem, selection)

        bucket = groups.setdefault(
            key,
            {"sound_stem": sound_stem, "selection": selection, "audios": {}, "plots": {}},
        )
        bucket["audios"][mode] = audio_path

    for plot_path in sorted(PLOTS_DIR.glob("*.png")):
        match = PLOT_FILE_PATTERN.match(plot_path.name)
        if not match:
            continue

        sound_stem = match.group("sound_stem")
        selection = int(match.group("selection"))
        mode = match.group("mode")
        plot_kind = match.group("plot_kind")
        key = (sound_stem, selection)

        bucket = groups.setdefault(
            key,
            {"sound_stem": sound_stem, "selection": selection, "audios": {}, "plots": {}},
        )
        bucket["plots"].setdefault(mode, {})[plot_kind] = plot_path

    return groups


def is_complete_group(group: dict) -> bool:
    required_plot_kinds = {"before", "after", "comparison", "noise_profile"}

    for mode in NOISE_MODE_ORDER:
        if mode not in group["audios"]:
            return False

        mode_plots = group["plots"].get(mode, {})
        if not required_plot_kinds.issubset(mode_plots.keys()):
            return False

    return True


def choose_output_group(groups: Dict[Tuple[str, int], dict]) -> Tuple[dict, int]:
    complete_groups = [group for group in groups.values() if is_complete_group(group)]
    if not complete_groups:
        raise FileNotFoundError(
            "No complete output set was found. Expected three cleaned WAV files "
            "and matching before/after/comparison/noise-profile plots."
        )

    complete_groups.sort(
        key=lambda group: (
            0 if group["selection"] == PREFERRED_SELECTION else 1,
            group["selection"],
            group["sound_stem"],
        )
    )
    return complete_groups[0], len(complete_groups)


def resolve_reference_spectrogram(row: pd.Series) -> Optional[Path]:
    if not truthy(row.get("spectrogram_exists", False)):
        return None

    candidate = Path(str(row["spectrogram_path"]))
    return candidate if candidate.exists() else None


@st.cache_data(show_spinner=False)
def load_demo_example() -> DemoExample:
    mapping_df = load_mapping_csv()
    output_groups = discover_output_groups()
    selected_group, complete_group_count = choose_output_group(output_groups)

    selection = int(selected_group["selection"])
    row = get_call_by_selection(selection, mapping_df)
    sound_file = str(row["Sound_file"])
    sound_stem = Path(sound_file).stem

    if sound_stem != selected_group["sound_stem"]:
        raise ValueError(
            "The saved output filenames do not match the mapping CSV for the selected example."
        )

    if not truthy(row.get("audio_exists", False)):
        raise FileNotFoundError(
            f"Mapped source audio is missing for selection {selection}: {row['audio_path']}"
        )

    source_audio_path = Path(str(row["audio_path"]))
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Source audio file not found: {source_audio_path}")

    signal, sample_rate = load_audio(source_audio_path)
    start_time = float(row["Start_time"])
    end_time = float(row["End_time"])
    crop_start = max(0.0, start_time - config.CROP_BUFFER_SEC)
    crop_end = end_time + config.CROP_BUFFER_SEC
    cropped_signal = crop_audio(signal, sample_rate, crop_start, crop_end)
    original_audio_bytes = to_audio_bytes(cropped_signal, sample_rate)

    modes: Dict[str, ModeArtifacts] = {}
    for mode in NOISE_MODE_ORDER:
        mode_plots = selected_group["plots"][mode]
        modes[mode] = ModeArtifacts(
            mode=mode,
            audio_path=selected_group["audios"][mode],
            before_plot_path=mode_plots["before"],
            after_plot_path=mode_plots["after"],
            comparison_plot_path=mode_plots["comparison"],
            noise_profile_plot_path=mode_plots["noise_profile"],
        )

    original_before_plot_path = modes["both"].before_plot_path

    return DemoExample(
        selection=selection,
        sound_file=sound_file,
        sound_stem=sound_stem,
        call_type=str(row["Call_type"]),
        start_time=start_time,
        end_time=end_time,
        crop_start=crop_start,
        crop_end=crop_end,
        source_audio_path=source_audio_path,
        reference_spectrogram_path=resolve_reference_spectrogram(row),
        original_before_plot_path=original_before_plot_path,
        original_audio_bytes=original_audio_bytes,
        modes=modes,
        available_group_count=complete_group_count,
    )


def render_metadata(example: DemoExample) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Selection", f"#{example.selection}")
    metric_columns[1].metric("Call Type", example.call_type.title())
    metric_columns[2].metric("Call Window", f"{example.start_time:.2f}s to {example.end_time:.2f}s")
    metric_columns[3].metric("Crop Window", f"{example.crop_start:.2f}s to {example.crop_end:.2f}s")
    metric_columns[4].metric("Cleaned Outputs", "3 modes")

    extra_copy = ""
    if example.available_group_count > 1:
        extra_copy = (
            f"<br>This UI is intentionally pinned to one example even though "
            f"{example.available_group_count} complete output sets were found."
        )

    st.markdown(
        f"""
        <div class="info-chip">
            Showing only <strong>{example.sound_file}</strong> from the current pipeline outputs.<br>
            Source audio: <strong>{example.source_audio_path}</strong><br>
            Saved artifacts are read directly from <strong>{OUTPUTS_DIR}</strong>.
            {extra_copy}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_original_section(example: DemoExample) -> None:
    st.markdown("<h2 class='section-title'>Original Recording</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='section-copy'>
            The original clip below uses the same buffered crop as the denoising pipeline,
            so it lines up with the saved before and after spectrograms.
        </p>
        """,
        unsafe_allow_html=True,
    )

    audio_player(
        example.original_audio_bytes,
        "Original cropped audio",
        "Buffered source clip used as the input to the cleanup pipeline.",
    )

    preview_columns = st.columns(2, gap="large")
    with preview_columns[0]:
        st.image(str(example.original_before_plot_path), use_container_width=True)
        st.caption("Pipeline spectrogram before cleaning")

    with preview_columns[1]:
        if example.reference_spectrogram_path is not None:
            st.image(str(example.reference_spectrogram_path), use_container_width=True)
            st.caption("Reference spectrogram linked from the mapping CSV")
        else:
            st.markdown(
                "<div class='empty-panel'>No reference spectrogram was found for this selection.</div>",
                unsafe_allow_html=True,
            )


def render_mode_section(artifacts: ModeArtifacts) -> None:
    st.markdown(
        f"<div class='mode-pill'>{NOISE_MODE_LABELS[artifacts.mode]}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h3 class='section-title' style='margin-top: 0;'>{NOISE_MODE_LABELS[artifacts.mode]}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p class='section-copy'>{NOISE_MODE_COPY[artifacts.mode]}</p>",
        unsafe_allow_html=True,
    )

    audio_player(
        artifacts.audio_path.read_bytes(),
        f"{NOISE_MODE_LABELS[artifacts.mode]} audio",
        "Saved cleaned WAV generated by the new pipeline.",
    )
    plot_tabs = st.tabs(["Before", "After", "Comparison", "Noise Profile", "3D Wave"])
    tab_items = [
        ("Before cleaning", artifacts.before_plot_path, "Spectrogram before denoising for this mode."),
        ("After cleaning", artifacts.after_plot_path, "Cleaned spectrogram produced by this mode."),
        ("Before vs after", artifacts.comparison_plot_path, "Side-by-side comparison for quick review."),
        ("Noise profile", artifacts.noise_profile_plot_path, "Estimated frequency-wise noise profile."),
    ]

    for tab, (_, image_path, caption) in zip(plot_tabs[:4], tab_items):
        with tab:
            st.image(str(image_path), use_container_width=True)
            st.caption(caption)

    with plot_tabs[4]:
        selected_perspective = st.radio(
            f"Wave model perspective for {artifacts.mode}",
            options=list(WAVE_PERSPECTIVES.keys()),
            format_func=lambda key: WAVE_PERSPECTIVES[key]["label"],
            horizontal=True,
            key=f"wave-perspective-{artifacts.mode}",
            label_visibility="collapsed",
        )
        st.caption("Inspect the cleaned recording as a futuristic 3D waveform from multiple perspectives.")

        wave_figure = build_wave_model_figure(str(artifacts.audio_path), selected_perspective)
        st.pyplot(wave_figure, use_container_width=True)
        plt.close(wave_figure)

    with st.expander("Saved files", expanded=False):
        files_df = pd.DataFrame(
            [
                ("Cleaned audio", str(artifacts.audio_path)),
                ("Before plot", str(artifacts.before_plot_path)),
                ("After plot", str(artifacts.after_plot_path)),
                ("Comparison plot", str(artifacts.comparison_plot_path)),
                ("Noise profile plot", str(artifacts.noise_profile_plot_path)),
            ],
            columns=["Artifact", "Path"],
        )
        st.table(files_df)


def main() -> None:
    inject_styles()
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Focused Demo</div>
            <h1>Elephant Noise Cleanup Output Review</h1>
            <p>
                This page is locked to one saved pipeline result. Instead of browsing many
                selections, it reads the existing files in <code>outputs/</code> and shows the
                original buffered clip plus the three cleaned variants for that same recording.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        example = load_demo_example()
    except Exception as exc:
        st.error(f"Unable to load the demo output: {exc}")
        return

    render_metadata(example)
    render_original_section(example)

    st.markdown("<h2 class='section-title'>Cleaned Outputs</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='section-copy'>
            Each section below shows one cleaned audio output for the same source recording,
            along with its saved spectrogram set from the current pipeline.
        </p>
        """,
        unsafe_allow_html=True,
    )

    for index, mode in enumerate(NOISE_MODE_ORDER):
        render_mode_section(example.modes[mode])
        if index < len(NOISE_MODE_ORDER) - 1:
            st.divider()


if __name__ == "__main__":
    main()

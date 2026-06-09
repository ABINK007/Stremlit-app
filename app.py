import streamlit as st
import pyaudio
import torch
import numpy as np
import os
import time
import threading
from datetime import datetime
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Live Transcription",
    page_icon="🎙️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme tweaks */
    .stApp { background-color: #0e1117; }

    .title-block {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .title-block h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .title-block p {
        color: #8b949e;
        font-size: 1rem;
        margin: 0.2rem 0 0 0;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .status-active {
        background: #1a3a2a;
        color: #3fb950;
        border: 1px solid #238636;
    }
    .status-idle {
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .status-speaking {
        background: #1f2d3d;
        color: #58a6ff;
        border: 1px solid #1f6feb;
    }

    /* Transcript box */
    .transcript-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        min-height: 320px;
        max-height: 520px;
        overflow-y: auto;
        font-family: 'Segoe UI', sans-serif;
    }
    .transcript-entry {
        margin-bottom: 0.9rem;
        padding: 0.6rem 0.8rem;
        background: #1c2128;
        border-left: 3px solid #238636;
        border-radius: 0 8px 8px 0;
    }
    .transcript-time {
        font-size: 0.72rem;
        color: #6e7681;
        margin-bottom: 3px;
    }
    .transcript-text {
        color: #e6edf3;
        font-size: 0.97rem;
        line-height: 1.5;
    }
    .transcript-meta {
        font-size: 0.7rem;
        color: #8b949e;
        margin-top: 3px;
    }
    .empty-state {
        color: #6e7681;
        text-align: center;
        padding: 3rem 1rem;
        font-size: 0.95rem;
    }

    /* VAD bar */
    .vad-bar-container {
        background: #21262d;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin-top: 4px;
    }
    .vad-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #238636, #3fb950);
        transition: width 0.1s ease;
    }

    /* Settings panel */
    .settings-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #8b949e !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, default in {
    "is_running": False,
    "entries": [],          # list of dicts: {time, text, elapsed, rms}
    "status": "idle",       # idle | listening | speaking | transcribing
    "rms_level": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────
p = pyaudio.PyAudio()

def list_input_devices():
    devices = []
    for idx in range(p.get_device_count()):
        info = p.get_device_info_by_index(idx)
        if info["maxInputChannels"] > 0:
            devices.append((idx, info["name"]))
    return devices


# ─────────────────────────────────────────────
# Whisper model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model, device

processor, model, DEVICE = load_whisper_model()
WHISPER_RATE = 16000


def transcribe(audio_chunk: np.ndarray, language: str, task: str) -> str:
    feats = processor(audio_chunk, sampling_rate=WHISPER_RATE, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats.to(DEVICE), language=language, task=task)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────
# VAD
# ─────────────────────────────────────────────
def compute_rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0


def vad_listen(
    device_index: int,
    language: str,
    task: str,
    vad_threshold: float,
    silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
):
    """
    Background thread: captures mic, detects speech via RMS VAD,
    collects speech segment, transcribes when silence detected.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = int(p.get_device_info_by_index(device_index)["defaultSampleRate"])
    FRAME_DURATION = 0.1        # seconds per read chunk (100ms)
    FRAME_SAMPLES = int(RATE * FRAME_DURATION)

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=FRAME_SAMPLES,
    )

    speech_frames: list[np.ndarray] = []
    silence_counter = 0.0
    in_speech = False
    speech_timer = 0.0

    st.session_state["status"] = "listening"

    while st.session_state["is_running"]:
        raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
        chunk = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        rms = compute_rms(chunk)
        st.session_state["rms_level"] = min(rms / (vad_threshold * 3), 1.0)  # normalised 0-1

        if rms > vad_threshold:
            # Speech detected
            speech_frames.append(chunk)
            silence_counter = 0.0
            speech_timer += FRAME_DURATION
            if not in_speech:
                in_speech = True
                st.session_state["status"] = "speaking"
            # Force flush if speech is very long
            if speech_timer >= max_speech_duration:
                _flush(speech_frames, RATE, language, task)
                speech_frames = []
                speech_timer = 0.0
                in_speech = False
                st.session_state["status"] = "listening"
        else:
            if in_speech:
                speech_frames.append(chunk)  # include trailing silence frame
                silence_counter += FRAME_DURATION
                if silence_counter >= silence_duration:
                    # End of utterance
                    if speech_timer >= min_speech_duration:
                        st.session_state["status"] = "transcribing"
                        _flush(speech_frames, RATE, language, task)
                    speech_frames = []
                    silence_counter = 0.0
                    speech_timer = 0.0
                    in_speech = False
                    st.session_state["status"] = "listening"

    stream.stop_stream()
    stream.close()
    st.session_state["status"] = "idle"
    st.session_state["is_running"] = False


def _flush(frames: list, rate: int, language: str, task: str):
    combined = np.concatenate(frames)
    resampled = resample(combined, int(len(combined) * WHISPER_RATE / rate))
    t0 = time.time()
    text = transcribe(resampled, language, task)
    elapsed = time.time() - t0
    if text:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": text,
            "elapsed": elapsed,
            "rms": compute_rms(combined),
        }
        st.session_state["entries"].append(entry)
        # Persist
        with open("transcriptions.txt", "a") as f:
            f.write(f"[{entry['time']}] {text}\n")


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🎙️ Live Audio Transcription</h1>
    <p>Automatic speech detection · Whisper ASR · No buttons needed</p>
</div>
""", unsafe_allow_html=True)

st.divider()

left_col, right_col = st.columns([1, 2], gap="large")

# ── Settings ──────────────────────────────────
with left_col:
    st.markdown("#### ⚙️ Settings")

    input_devices = list_input_devices()
    device_names = [name for _, name in input_devices]
    device_indices = [idx for idx, _ in input_devices]

    selected_name = st.selectbox("🎤 Microphone", device_names, disabled=st.session_state["is_running"])
    selected_index = device_indices[device_names.index(selected_name)]

    language = st.selectbox(
        "🌐 Audio Language",
        ["English", "Hindi", "French", "German", "Spanish", "Japanese", "Chinese"],
        disabled=st.session_state["is_running"],
    )
    task = st.selectbox(
        "📝 Task",
        ["transcribe", "translate"],
        disabled=st.session_state["is_running"],
        help="'translate' outputs English regardless of source language",
    )
    if task == "translate":
        st.caption("Translation → output will be in **English**")

    st.markdown("---")
    st.markdown("##### 🔊 VAD Settings")

    vad_threshold = st.slider(
        "Silence threshold (RMS)",
        min_value=0.005, max_value=0.08, value=0.02, step=0.005,
        format="%.3f",
        disabled=st.session_state["is_running"],
        help="Lower = more sensitive. Raise if background noise triggers transcription.",
    )
    silence_dur = st.slider(
        "Silence to end utterance (s)",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        disabled=st.session_state["is_running"],
    )
    min_speech = st.slider(
        "Min speech duration (s)",
        min_value=0.3, max_value=2.0, value=0.5, step=0.1,
        disabled=st.session_state["is_running"],
        help="Chunks shorter than this are discarded (prevents noise pops).",
    )
    max_speech = st.slider(
        "Max speech duration (s)",
        min_value=5, max_value=60, value=30, step=5,
        disabled=st.session_state["is_running"],
        help="Force transcription if utterance is very long.",
    )

    st.markdown("---")

    # Toggle button
    if not st.session_state["is_running"]:
        if st.button("▶️ Start Listening", use_container_width=True, type="primary"):
            st.session_state["is_running"] = True
            st.session_state["entries"] = []
            if os.path.exists("transcriptions.txt"):
                os.remove("transcriptions.txt")
            t = threading.Thread(
                target=vad_listen,
                args=(selected_index, language, task, vad_threshold, silence_dur, min_speech, max_speech),
                daemon=True,
            )
            t.start()
            st.rerun()
    else:
        if st.button("⏹️ Stop Listening", use_container_width=True):
            st.session_state["is_running"] = False
            st.rerun()

    # Status indicator
    status = st.session_state["status"]
    status_map = {
        "idle": ("status-idle", "⚫ Idle"),
        "listening": ("status-active", "🟢 Listening…"),
        "speaking": ("status-speaking", "🔵 Speech detected"),
        "transcribing": ("status-speaking", "⚡ Transcribing…"),
    }
    cls, label = status_map.get(status, ("status-idle", "⚫ Idle"))
    st.markdown(f'<div class="status-pill {cls}">{label}</div>', unsafe_allow_html=True)

    # RMS level bar
    if st.session_state["is_running"]:
        rms_pct = int(st.session_state["rms_level"] * 100)
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#6e7681; margin-top:8px">🔉 Mic level</div>
        <div class="vad-bar-container">
            <div class="vad-bar-fill" style="width:{rms_pct}%"></div>
        </div>
        """, unsafe_allow_html=True)


# ── Transcript ────────────────────────────────
with right_col:
    entries = st.session_state["entries"]
    n = len(entries)

    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"#### 📄 Transcript  <span style='font-size:0.8rem; color:#6e7681'>({n} segment{'s' if n != 1 else ''})</span>", unsafe_allow_html=True)
    with clear_col:
        if st.button("🗑️ Clear", disabled=(n == 0)):
            st.session_state["entries"] = []
            st.rerun()

    if not entries:
        st.markdown("""
        <div class="transcript-container">
            <div class="empty-state">
                🎙️ Start listening and speak — transcription appears here automatically.<br>
                <span style="font-size:0.82rem">No buttons required</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        html_entries = ""
        for e in reversed(entries):  # newest first
            html_entries += f"""
            <div class="transcript-entry">
                <div class="transcript-time">{e['time']}</div>
                <div class="transcript-text">{e['text']}</div>
                <div class="transcript-meta">⏱ {e['elapsed']:.2f}s inference</div>
            </div>
            """
        st.markdown(f'<div class="transcript-container">{html_entries}</div>', unsafe_allow_html=True)

    # Download button
    if entries:
        full_text = "\n".join(f"[{e['time']}] {e['text']}" for e in entries)
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=full_text,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

# ─────────────────────────────────────────────
# Auto-refresh while running
# ─────────────────────────────────────────────
if st.session_state["is_running"]:
    time.sleep(0.8)
    st.rerun()

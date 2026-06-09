import streamlit as st
import av
import queue
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Live Audio Transcription",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Live Audio Transcription")
st.write("Speak into your microphone and see live transcription.")

# ----------------------------------
# Load Whisper
# ----------------------------------

@st.cache_resource
def load_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

model = load_model()

# ----------------------------------
# Audio Queue
# ----------------------------------

audio_queue = queue.Queue()

# ----------------------------------
# Audio Processor
# ----------------------------------

class AudioProcessor:

    def recv(self, frame):

        audio = frame.to_ndarray()

        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        audio_queue.put(audio.astype(np.float32))

        return frame


# ----------------------------------
# WebRTC Microphone
# ----------------------------------

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
    audio_processor_factory=AudioProcessor,
)

# ----------------------------------
# Session State
# ----------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = []

# ----------------------------------
# Process Audio
# ----------------------------------

if ctx.state.playing:

    st.success("🎤 Microphone Active")

    if not audio_queue.empty():

        audio_chunks = []

        while not audio_queue.empty():
            audio_chunks.append(audio_queue.get())

        audio_data = np.concatenate(audio_chunks)

        if len(audio_data) > 16000:

            segments, _ = model.transcribe(
                audio_data,
                language="en",
                vad_filter=True
            )

            text = " ".join(
                segment.text
                for segment in segments
            ).strip()

            if text:

                st.session_state.transcript.append(text)

# ----------------------------------
# Transcript Display
# ----------------------------------

st.subheader("Transcript")

for line in reversed(st.session_state.transcript):
    st.write(line)

# ----------------------------------
# Download
# ----------------------------------

if st.session_state.transcript:

    full_text = "\n".join(st.session_state.transcript)

    st.download_button(
        "Download Transcript",
        full_text,
        file_name="transcript.txt"
    )
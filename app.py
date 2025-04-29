import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode
import numpy as np
import whisper
from openai import OpenAI
import os
import tempfile
import soundfile as sf
from pathlib import Path
import base64
import requests
from urllib.parse import quote

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
TAVUS_API_KEY = st.secrets["TAVUS_API_KEY"]

# App configuration
st.set_page_config(page_title="Project Neon", page_icon="🎙️", layout="wide")

# Constants
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

# ====== Video Conversation Functions ======
def start_neon_conversation():
    """Start Project Neon video conversation and return URL"""
    neon_context = """
    You are Project Neon - an advanced AI assistant that:
    1. Provides real-time access to enterprise knowledge systems
    2. Answers questions using connected databases (SQL, Postgres, Databricks)
    3. Retrieves information from vector stores of unstructured data
    4. Delivers responses in clear, conversational language
    """
    
    payload = {
        "replica_id": "r79e1c033f",
        "persona_id": "p9a95912",
        "conversation_name": "Project Neon Session",
        "conversational_context": neon_context.strip(),
        "custom_greeting": "Hello! I'm Neon, your AI knowledge assistant. What would you like to know?",
        "properties": {
            "max_call_duration": 1800,
            "enable_transcription": True,
            "language": "english"
        }
    }
    
    headers = {
        "x-api-key": TAVUS_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://tavusapi.com/v2/conversations",
            json=payload,
            headers=headers,
            timeout=10
        )
        return response.json().get("conversation_url") if response.ok else None
    except Exception as e:
        st.error(f"Error starting conversation: {str(e)}")
        return None

# ====== Voice Assistant Functions ======
def autoplay_video(video_path):
    """Auto-play video from URL (optimized for Streamlit Cloud)"""
    try:
        # Option 1: Hosted on GitHub (recommended)
        video_url = "project_neon_intro.mp4"
        st.video(video_url, autoplay=True, format="video/mp4")
        
    except Exception as e:
        st.error(f"Video failed to load: {str(e)}")

def record_audio():
    """Record audio using streamlit-webrtc"""
    audio_frames = []
    ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={
            "audio": True,
            "video": False
        }
    )
    
    if ctx.audio_receiver:
        try:
            for frame in ctx.audio_receiver.get_frames(timeout=RECORD_SECONDS):
                audio_frames.append(frame.to_ndarray())
            if audio_frames:
                return np.concatenate(audio_frames)
        except Exception as e:
            st.error(f"Recording error: {e}")
    return None

def transcribe_audio(audio_data):
    """Transcribe audio using Whisper"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            scaled = np.int16(audio_data * 32767)
            import scipy.io.wavfile
            scipy.io.wavfile.write(tmp_path, SAMPLE_RATE, scaled)
        
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        
        try:
            os.unlink(tmp_path)
        except PermissionError:
            import atexit
            atexit.register(lambda: os.unlink(tmp_path) if os.path.exists(tmp_path) else None)
        
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def text_to_speech(text):
    """Convert text to speech"""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    return response.content

def play_audio(audio_bytes):
    """Play audio in browser"""
    st.audio(audio_bytes, format="audio/mp3")

# ====== Main App Layout ======
def main():
    st.title("Project Neon - AI Assistant")
    st.markdown("---")
    # Initialize session state
    if 'voice_state' not in st.session_state:
        st.session_state.voice_state = {
            'recording': False,
            'processing': False,
            'user_text': None,
            'gpt_response': None
        }
    
    # Introduction video section
    video_path = "project_neon_intro.mp4"
    if Path(video_path).exists():
        autoplay_video(video_path)
    else:
        st.warning("Introduction video not found. Place your 'project_neon_intro.mp4' in the same directory.")
    
    st.markdown("---")
    
    # Create two columns for interaction options
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎥 Video Conversation")
        st.markdown("Have a face-to-face conversation with Neon")
        if st.button("Start Video Chat", key="video_btn", type="primary"):
            with st.spinner("Connecting to Neon..."):
                conversation_url = start_neon_conversation()
            
            if conversation_url:
                st.success("Connection established!")
                st.markdown(f"""
                    <a href="{conversation_url}" target="_blank">
                        <button style="
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 20px;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 16px;
                            width: 100%;
                        ">
                            ➤ Enter Video Conversation
                        </button>
                    </a>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to start conversation. Please try again.")
                
    with col2:
        st.header("🎙️ Voice Assistant")
        st.markdown("Speak with Neon using your microphone")
        
        if st.button("Start Recording", disabled=st.session_state.voice_state['processing']):
            st.session_state.voice_state['recording'] = True
            st.session_state.voice_state['processing'] = True
            st.experimental_rerun()
        
        if st.session_state.voice_state['recording']:
            with st.spinner("Recording... Speak now!"):
                audio_data = record_audio()
                if audio_data is not None:
                    st.session_state.voice_state['user_text'] = transcribe_audio(audio_data)
                st.session_state.voice_state['recording'] = False
                st.experimental_rerun()
        
        if st.session_state.voice_state['user_text']:
            st.write(f"**You said:** {st.session_state.voice_state['user_text']}")
            
            if st.session_state.voice_state['gpt_response'] is None:
                with st.spinner("Processing your request..."):
                    st.session_state.voice_state['gpt_response'] = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are Project Neon..."},
                            {"role": "user", "content": st.session_state.voice_state['user_text']}
                        ]
                    ).choices[0].message.content
                st.experimental_rerun()
            
            st.write(f"**Neon responds:** {st.session_state.voice_state['gpt_response']}")
            
            if st.button("Play Response"):
                audio_response = text_to_speech(st.session_state.voice_state['gpt_response'])
                play_audio(audio_response)
            
            if st.button("Start New Conversation"):
                st.session_state.voice_state = {
                    'recording': False,
                    'processing': False,
                    'user_text': None,
                    'gpt_response': None
                }
                st.experimental_rerun()

if __name__ == "__main__":
    main()
    

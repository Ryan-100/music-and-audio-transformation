import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import convolve
import io
import librosa
import requests
from streamlit_lottie import st_lottie
import time

# Function to load Lottie animations from a URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set the page configuration
st.set_page_config(
    page_title="Music and Sound Transformation",
    page_icon="🎵",
    layout="wide"
)

# --- Custom CSS for Gradient Text and Background ---
st.markdown("""
<style>
/* Animated Gradient Title */
.gradient-text {
    background: -webkit-linear-gradient(45deg, #ff7e5f, #feb47b, #86A8E7, #91EAE4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-animation 8s ease infinite;
    background-size: 400% 400%;
}
@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Lively Gradient Background for the main app area */
[data-testid="stAppViewContainer"] > .main {
    background-image: linear-gradient(to right, #ff7e5f, #feb47b);
}
</style>
""", unsafe_allow_html=True)


# --- Main App Logic ---
def main():
    st.markdown('<h1 style="text-align: center;" class="gradient-text">🎵 Music and Sound Transformation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Transforming audio by treating it as a numerical vector.</p>', unsafe_allow_html=True)
    st.divider()

    # Load Lottie Animations
    lottie_original = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_pGwn4p.json")
    lottie_transformed = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_bwnh5s.json")

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("🎛️ Controls")
        uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "flac", "mp3"])
        st.divider()

        if uploaded_file:
            # Controls are only shown after a file is uploaded
            st.header("Voice Changer")
            voice_choice = st.selectbox("Pitch Effect", ("None", "Chipmunk Voice", "Child Voice", "Female Voice", "Deep Voice", "Monster Voice"))
            st.divider()
            st.header("Vector Operations")
            scale_factor = st.slider("Scaling (Amplitude)", 0.0, 2.0, 1.0, 0.1)
            is_reflected = st.checkbox("Reflect (Invert Phase)")
            is_reversed = st.checkbox("Reverse (Time Reversal)")
            st.divider()
            st.header("Effects")
            filter_size = st.slider("Filter (Smoothing)", 1, 100, 1, 1)

    # --- Main Content Area ---
    if uploaded_file is not None:
        # This section runs when a file IS uploaded
        data, rate = sf.read(io.BytesIO(uploaded_file.getvalue()))
        data = data.astype(np.float32)
        if data.ndim > 1: data = data[:, 0]
            
        col1, col2 = st.columns(2)

        with col1:
            if lottie_original: st_lottie(lottie_original, speed=1, height=150, key="original")
            st.subheader("✨ Original Audio")
            st.audio(uploaded_file)
        
        with st.spinner('Applying transformations... please wait.'):
            transformed_data = data.copy()
            if voice_choice != "None":
                n_steps = {"Chipmunk Voice": 10.0, "Child Voice": 6.0, "Female Voice": 3.0, "Deep Voice": -4.0, "Monster Voice": -8.0}.get(voice_choice, 0.0)
                transformed_data = librosa.effects.pitch_shift(y=transformed_data, sr=rate, n_steps=n_steps)
            if is_reversed: transformed_data = transformed_data[::-1]
            final_scale = -scale_factor if is_reflected else scale_factor
            transformed_data = transformed_data * final_scale
            if filter_size > 1:
                kernel = np.ones(filter_size) / filter_size
                transformed_data = convolve(transformed_data, kernel, mode="same")
            
            virtual_file = io.BytesIO()
            sf.write(virtual_file, transformed_data, rate, format='WAV')
            virtual_file.seek(0)
        
        with col2:
            if lottie_transformed: st_lottie(lottie_transformed, speed=1, height=150, key="transformed")
            st.subheader("✨ Transformed Audio")
            st.audio(virtual_file, format='audio/wav')
            st.download_button("📥 Download Transformed WAV", virtual_file, "transformed_audio.wav", "audio/wav")
            
        st.divider()
        st.markdown('<p style="text-align: center; font-weight: bold;" class="gradient-text">"Where words fail, music speaks."</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-weight: bold;" class="gradient-text">"Music can change the world."</p>', unsafe_allow_html=True)

    else:
        # This is the stylish "empty state" that shows when no file is uploaded
        lottie_welcome = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_q23fgt.json")
        if lottie_welcome:
            st_lottie(lottie_welcome, speed=1, height=200, key="welcome")
            
        st.header("Get Started in 3 Simple Steps")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center;'>📤</h3>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>1. Upload File</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Use the sidebar to upload your audio file (WAV, FLAC, or MP3).</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h3 style='text-align: center;'>🎛️</h3>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>2. Apply Effects</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Play with the controls to change pitch, reverse the audio, and more.</p>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h3 style='text-align: center;'>📥</h3>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>3. Download</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Listen to your new creation and download it as a WAV file.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
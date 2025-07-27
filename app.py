import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Mental Health App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS eksternal
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style/style.css")

# Fungsi konversi gambar ke base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load ilustrasi gambar
image = Image.open("assets/images/illustration1.png")
img_base64 = image_to_base64(image)

st.markdown(f"""
<div class="page-wrapper">
    <div class="left-content">
        <h1 class="title-main">
            Healthy Minds, Happy Lives. So, Take Care Of Ur <span class="blue-text">Mental Health</span>
        </h1>
        <p class="quote">
            “Bukan dia, bukan aku, bukan kamu, dan bukan siapapun, hanya suara kecil di kepala yang bilang dirimu ga cukup, cegah suara itu teriak lebih keras.
            Mau kamu main character, side character, atau anti-hero jangan biarkan lukamu menjadi musuh yang sulit ditaklukan.<br>
            Spreading love, you're enough babe.”
        </p>
        <a href="/tentang" target="_self" class="custom-button">Jelajahi Dunia MentalCare</a>
    </div>
    <div class="right-content">
        <img src="data:image/png;base64,{img_base64}" class="illustration"/>
    </div>
</div>
""", unsafe_allow_html=True)
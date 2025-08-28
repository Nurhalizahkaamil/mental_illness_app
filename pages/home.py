import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import json
import pathlib
from datetime import datetime, timezone, timedelta
import locale
import re  # 🔹 Untuk regex highlight

from utils.twitter_scraper import get_recent_tweets_with_cache
from utils.predict import predict_user_and_save
from utils.database import Database

# ─────────────────────────────────────────────
# Halaman dan CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Identifikasi Mental Health", layout="wide")

css_path = pathlib.Path("style_home.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
div.stButton > button {
    background-color: #2f70c6 !important;
    color: white !important;
    border: 2px solid #2f70c6 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load ilustrasi default
# ─────────────────────────────────────────────
image = Image.open("assets/images/illustration2.png")
buffered = BytesIO()
image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# ─────────────────────────────────────────────
# Load label mapping
# ─────────────────────────────────────────────
try:
    with open("models/label_mapping_lvl2.json", "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
except Exception as e:
    st.error(f"Gagal memuat label mapping: {e}")
    st.stop()

# 🔹 Load keyword gangguan mental dari JSON
try:
    with open("models/gangguan_keyword.json", "r", encoding="utf-8") as f:
        keyword_map = json.load(f)
except Exception as e:
    st.error(f"Gagal memuat keyword gangguan: {e}")
    keyword_map = {}

# 🔹 Fungsi highlight kata kunci
def highlight_keywords(text, keyword_map):
    for _, keywords in keyword_map.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            text = re.sub(pattern, f"**{kw}**", text, flags=re.IGNORECASE)
    return text

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
try:
    db = Database()
except Exception as e:
    st.error(f"Gagal koneksi ke database: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Locale untuk tanggal Indonesia
# ─────────────────────────────────────────────
try:
    locale.setlocale(locale.LC_TIME, "id_ID.UTF-8")
except:
    try:
        locale.setlocale(locale.LC_TIME, "Indonesian_indonesia")
    except:
        locale.setlocale(locale.LC_TIME, "")

# ─────────────────────────────────────────────
# Variabel awal
# ─────────────────────────────────────────────
result = None

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("Ayo Mulai Identifikasi!")
    username_input = st.text_input("Masukkan Username X kamu")

    if st.button("Prediksi Sekarang"):
        username = username_input.strip().replace("@", "").lower()

        if not username:
            st.warning("Silakan masukkan username terlebih dahulu.")
        else:
            tweets = get_recent_tweets_with_cache(
                username=username,
                force_refresh=True,
                limit=10
            )

            if tweets is None:
                st.error("Gagal mengambil tweet. Username mungkin tidak valid.")
                st.link_button("Isi Manual Postingan", "./manual_input")
            elif isinstance(tweets, str) or not tweets:
                st.warning("Tidak ada tweet yang bisa dianalisis.")
                st.link_button("Isi Manual Postingan", "./manual_input")
            else:
                with st.spinner("Sedang menganalisis tweet..."):
                    result = predict_user_and_save(username, tweets, db)

                st.success("Identifikasi berhasil dilakukan.")
                st.markdown("### Berikut Detail Klasifikasi Akunmu:")

                label_lv1 = result.get("label_lv1", "none")
                label_lv2 = result.get("label_lv2", "-")

                if label_lv1 == "none":
                    st.info("Belum ditemukan indikasi kuat dari pola psikologis yang mengkhawatirkan.")
                    st.write("**Manifestasi psikologis:** Tidak ada")
                else:
                    st.error("Ditemukan indikasi kondisi psikologis.")
                    st.write(f"**Manifestasi psikologis:** {label_lv2.capitalize()}")
                    st.markdown("---")
                    st.link_button(
                        "Lihat Detail Manifestasi kamu disini",
                        url=f"/detail?gangguan={label_lv2.lower()}"
                    )

# ─────────────────────────────────────────────
# Tampilkan Tweet + Tanggal + Highlight Kata Kunci
# ─────────────────────────────────────────────
if result and result.get("tweets"):
    with st.expander("Lihat Postingan yang Dianalisis"):
        WIB = timezone(timedelta(hours=7))  # UTC+7

        for i, tweet in enumerate(result["tweets"], 1):
            tipe = tweet.get("type", "-").capitalize() if isinstance(tweet, dict) else "Tweet"
            tanggal = tweet.get("created_at", "-") if isinstance(tweet, dict) else "-"

            # Format tanggal → human readable (Bahasa Indonesia + WIB)
            try:
                if "T" in tanggal:  # Format ISO 8601
                    tanggal_obj = datetime.fromisoformat(tanggal.replace("Z", "+00:00")).astimezone(WIB)
                else:  # Format custom dari API
                    tanggal_obj = datetime.strptime(tanggal, "%Y-%m-%d %H:%M:%S WIB").replace(tzinfo=WIB)

                tanggal_str = tanggal_obj.strftime("%A, %d %B %Y pukul %H:%M WIB")
            except Exception:
                tanggal_str = tanggal

            text = tweet.get("text", tweet) if isinstance(tweet, dict) else tweet
            if len(text) > 300:
                text = text[:300] + "..."

            # 🔹 Highlight kata kunci di teks tweet
            highlighted_text = highlight_keywords(text, keyword_map)

            st.markdown(f"**Tweet {i}** ({tipe})  \n**{tanggal_str}**  \n{highlighted_text}")

with col2:
    st.markdown(f"""
    <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" />
    """, unsafe_allow_html=True)

db.close()

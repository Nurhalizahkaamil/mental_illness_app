import streamlit as st
from utils.predict import predict_user_and_save
from utils.database import Database
import pathlib
import json
import re  # 🔹 import regex untuk highlight

# ─────────────────────────────────────────────
# Setup halaman dan style
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Input Manual Cerita",
    layout="centered",
    initial_sidebar_state="collapsed"
)

css_path = pathlib.Path("assets/style/style_manual.css")
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
# Load keyword gangguan mental untuk highlight
# ─────────────────────────────────────────────
try:
    with open("models/gangguan_keyword.json", "r", encoding="utf-8") as f:
        keyword_map = json.load(f)
except Exception as e:
    st.error(f"Gagal memuat keyword gangguan: {e}")
    keyword_map = {}

def highlight_keywords(text, keyword_map):
    for _, keywords in keyword_map.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            text = re.sub(pattern, f"**{kw}**", text, flags=re.IGNORECASE)
    return text

# ─────────────────────────────────────────────
# Koneksi database
# ─────────────────────────────────────────────
try:
    db = Database()
except Exception as e:
    st.error(f"❌ Gagal koneksi ke database: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Form input manual (versi cerita)
# ─────────────────────────────────────────────
st.markdown("## Yuk Ceritakan Perasaanmu 💬")
st.markdown("Ceritakan apa yang kamu rasakan beberapa hari belakangan ini secara bebas dan jujur. Bisa soal suasana hati, kejadian yang bikin sedih atau senang, atau apapun yang mengganggu pikiranmu akhir-akhir ini.")

with st.form("manual_form"):
    story = st.text_area(
        "Ceritakan perasaanmu di sini ✨",
        placeholder="Contoh: Akhir-akhir ini aku merasa cepat marah dan mudah lelah. Kadang merasa cemas tanpa alasan, dan jadi sering menjauh dari teman-teman...",
        height=250,
        key="story_input"
    )

    submitted = st.form_submit_button("Prediksi Ceritamu")

    if submitted:
        story_clean = story.strip()

        if not story_clean:
            st.warning("Harap isi cerita kamu terlebih dahulu sebelum melanjutkan.")
            st.stop()

        # Validasi minimal 6 kata
        word_count = len(story_clean.split())
        if word_count < 6:
            st.warning(f"Ceritamu terlalu pendek ({word_count} kata). Coba tambahkan sedikit lagi (minimal 6 kata) agar bisa dianalisis.")
            st.stop()

        # Potong menjadi maksimal 5 kalimat jika terlalu panjang
        all_posts_clean = story_clean.split(".")
        all_posts_clean = [p.strip() for p in all_posts_clean if p.strip()]
        if len(all_posts_clean) > 5:
            all_posts_clean = all_posts_clean[:5]

        with st.spinner("Sedang menganalisis..."):
            result = predict_user_and_save(username="anon", tweets=all_posts_clean, db=db)

        st.success("Prediksi berhasil dilakukan!")

        # 🔹 Tampilkan cerita dengan highlight kata kunci
        highlighted_story = highlight_keywords(story_clean, keyword_map)
        st.markdown("### Cerita yang kamu tulis:")
        st.markdown(highlighted_story)

        if result["final_status"] == "terindikasi":
            st.error("**Ditemukan pola yang mengarah pada indikasi kondisi psikologis tertentu.**")
            st.write(f"**Manifestasi psikologis yang terdeteksi:** {result['label_lv2'].capitalize()}")
            st.markdown("---")
            st.link_button(
                "Lihat Detail Manifestasi kamu disini",
                url=f"/detail?gangguan={result['label_lv2'].lower()}"
            )
        else:
            st.info("**Wow! Belum ditemukan indikasi kuat dari pola psikologis yang mengkhawatirkan. Terus jaga kesehatan mentalmu, ya!**")
            st.write("**Manifestasi psikologis:** Tidak ada")

db.close()

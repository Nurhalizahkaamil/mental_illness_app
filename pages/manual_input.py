import streamlit as st
from utils.predict import predict_user_and_save
from utils.database import Database
import pathlib

# ─────────────────────────────────────────────
# Setup halaman dan style
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Input Manual Postingan",
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
# Koneksi database
# ─────────────────────────────────────────────
try:
    db = Database()
except Exception as e:
    st.error(f"❌ Gagal koneksi ke database: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Form input manual
# ─────────────────────────────────────────────
st.markdown("## SILAHKAN MASUKAN 5 KALIMAT KAMU DISINI YAA")

with st.form("manual_form"):
    username = st.text_input("Masukkan username Twitter (tanpa @)/ Namamu disini", key="username_input")
    post1 = st.text_input("Kalimat 1", key="post1_input")
    post2 = st.text_input("Kalimat 2", key="post2_input")
    post3 = st.text_input("Kalimat 3", key="post3_input")
    post4 = st.text_input("Kalimat 4", key="post4_input")
    post5 = st.text_input("Kalimat 5", key="post5_input")

    submitted = st.form_submit_button("Prediksi Kalimatmu")

    if submitted:
        username = username.strip()
        all_posts = [post1, post2, post3, post4, post5]
        all_posts_clean = [p.strip() for p in all_posts if p.strip()]

        if not username or len(all_posts_clean) != 5:
            st.warning("Harap isi form dengan lengkap terlebih dahulu sebelum melanjutkan.")
        else:
            with st.spinner("Sedang menganalisis..."):
                result = predict_user_and_save(username=username, tweets=all_posts_clean, db=db)

            st.success("Prediksi berhasil dilakukan!")

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

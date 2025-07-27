# pages/detail.py

import streamlit as st
from urllib.parse import unquote
from pages.visualizer import show_gangguan_card

st.set_page_config(
    page_title="Detail Manifestasi Psikologis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ambil parameter dari URL dengan cara baru (tanpa warning)
query_params = st.query_params
gangguan = unquote(query_params.get("gangguan", "")).lower().strip()

st.markdown("## Detail Manifestasi Psikologis Kamu")

if gangguan:
    show_gangguan_card(label_lv2=gangguan)
else:
    st.warning("Tidak ada gangguan yang dipilih.")

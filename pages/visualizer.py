import streamlit as st
import json
from PIL import Image
import base64
from io import BytesIO
import os

def show_gangguan_card(label_lv2: str, 
                        image_path: str = "assets/images/illustration3.png", 
                        mapping_path: str = "models/gangguan_deskripsi.json"):
    """
    Menampilkan hasil identifikasi psikologis berdasarkan label_lv2 dalam layout 2 kolom seperti gambar HASIL.png
    """
    try:
        # Pastikan file JSON tersedia
        if not os.path.exists(mapping_path):
            st.warning("File deskripsi gangguan tidak ditemukan.")
            return

        with open(mapping_path, "r", encoding="utf-8") as f:
            deskripsi_map = json.load(f)

        data = deskripsi_map.get(label_lv2.lower(), None)

        if not data:
            st.warning("Deskripsi untuk gangguan ini belum tersedia.")
            return

        # Encode gambar ke base64
        if os.path.exists(image_path):
            image = Image.open(image_path)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
        else:
            img_base64 = None

        # Layout 2 kolom
        col1, col2 = st.columns([1, 1.4])

        with col1:
            if img_base64:
                st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto;"/>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Gambar tidak tersedia.")

        with col2:
            # Judul & deskripsi
            st.markdown(f"## {data['judul']}")
            st.write(data['deskripsi'])
            st.write(data['motivasi'])

            # Gejala umum
            st.markdown("#### Gejala Umum")
            st.markdown(f"<pre style='font-size: 15px'>{data['gejala']}</pre>", unsafe_allow_html=True)

            # Saran & solusi
            st.markdown("#### Saran & Solusi")
            st.write(data['solusi'])

            # Tambahkan link referensi jika tersedia
            if "link_referensi" in data and data["link_referensi"]:
                st.markdown(
                    f"<p style='margin-top:1rem;'>Pelajari dan kenali lebih lanjut kondisi psikologismu "
                    f"<a href='{data['link_referensi']}' target='_blank' style='color:#1E88E5; font-weight:bold;'>di sini</a>.</p>",
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Gagal memuat informasi gangguan: {e}")

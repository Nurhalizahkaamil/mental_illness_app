import streamlit as st
import matplotlib.pyplot as plt
from utils.database import Database

# Konfigurasi halaman
st.set_page_config(
    page_title="Tentang Aplikasi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk wave background CSS
def set_wave_background():
    st.markdown("""
        <style>
        .main .block-container {
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }
        .wave-container {
            width: 100vw;
            margin-left: calc(-50vw + 50%);
            background: #DAEDFF;
            padding: 5rem 2rem 10rem 2rem;
            position: relative;
            color: #1f3c88;
        }
        .wave-container::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 80px;
            background: white;
            clip-path: ellipse(60% 100% at 50% 100%);
        }
        html, body, [class*="main"] {
            height: 100%;
            overflow-x: hidden;
        }
        section, .block-container {
            min-height: 100vh;
            box-sizing: border-box;
        }

        /* Styling tombol custom-button */
        .custom-button {
            background-color: #2f70c6;
            padding: 0.7rem 1.5rem;
            border-radius: 30px;
            color: white !important;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)

# === DATA GANGGUAN ===
gangguan_data = {
    "anxiety": {
        "judul": "Anxiety Disorder",
        "deskripsi": (
            "Gangguan kecemasan (anxiety disorder) adalah kondisi psikologis yang ditandai dengan "
            "kekhawatiran berlebihan terhadap berbagai hal, meskipun tidak ada ancaman nyata. "
            "Kecemasan bisa muncul dalam bentuk serangan panik, fobia spesifik, atau kecemasan sosial. "
            "Kondisi ini bisa mengganggu aktivitas sehari-hari dan membuat seseorang merasa tegang atau lelah secara terus-menerus."
        ),
        "gejala": (
            "Perasaan gelisah terus-menerus, Ketegangan otot, Sulit tidur, Detak jantung cepat, "
            "Sulit berkonsentrasi, Perasaan panik tanpa alasan yang jelas."
        ),
        "solusi": (
            "Mulailah dengan teknik pernapasan dalam atau meditasi singkat setiap hari. "
            "Aktivitas fisik ringan seperti berjalan kaki dapat membantu menenangkan pikiran. "
            "Cobalah menulis jurnal untuk memahami pola kecemasanmu dan mengidentifikasi pemicu tertentu."
        ),
        "terapi": (
            "Untuk kasus sedang hingga berat, terapi perilaku kognitif (CBT) sangat disarankan. "
            "Terapi ini membantu mengubah pola pikir negatif menjadi lebih adaptif. "
            "Psikiater juga dapat meresepkan obat seperti antidepresan (SSRI) atau anti-kecemasan jangka pendek jika diperlukan."
        )
    },

    "depression": {
        "judul": "Depression (Depresi)",
        "deskripsi": (
            "Depresi adalah gangguan suasana hati yang membuat seseorang merasa sedih berkepanjangan, kehilangan minat, dan kehilangan energi. "
            "Ini bukan sekadar 'sedih biasa' dan dapat berdampak pada hubungan sosial, pekerjaan, dan kesehatan fisik. "
            "Depresi bisa dipicu oleh stres berkepanjangan, trauma masa lalu, ketidakseimbangan kimia otak, atau faktor genetik."
        ),
        "gejala": (
            "Perasaan sedih mendalam, Kehilangan semangat dan motivasi, Gangguan tidur atau tidur berlebihan, "
            "Merasa tidak berharga, Mudah lelah, Pikiran untuk menarik diri atau menyakiti diri sendiri."
        ),
        "solusi": (
            "Coba buat rutinitas sederhana yang menyenangkan, seperti jalan pagi atau mendengarkan musik. "
            "Berbagi cerita dengan seseorang yang kamu percaya juga dapat meringankan beban emosional. "
            "Hindari mengisolasi diri, dan cari lingkungan yang mendukung proses pemulihanmu."
        ),
        "terapi": (
            "Psikoterapi seperti CBT atau terapi interpersonal bisa sangat membantu dalam mengatasi depresi. "
            "Dalam beberapa kasus, dokter dapat meresepkan antidepresan seperti SSRI atau SNRI untuk menstabilkan mood. "
            "Terapi kombinasi (psikolog dan psikiater) sering kali lebih efektif daripada satu pendekatan saja."
        )
    },

    "bipolar": {
        "judul": "Bipolar Disorder",
        "deskripsi": (
            "Bipolar adalah gangguan suasana hati yang ditandai oleh fluktuasi ekstrem antara periode mania (energi tinggi, impulsif, merasa sangat bahagia) dan depresi (kesedihan, kelelahan, rasa putus asa). "
            "Perubahan ini bisa berlangsung selama berhari-hari atau bahkan berminggu-minggu dan memengaruhi pekerjaan, hubungan, dan kesehatan secara menyeluruh."
        ),
        "gejala": (
            "Fase mania: berbicara sangat cepat, kepercayaan diri berlebihan, ide besar, kebutuhan tidur menurun. "
            "Fase depresi: energi sangat rendah, kesulitan fokus, perasaan tidak berdaya, keinginan menarik diri."
        ),
        "solusi": (
            "Buat catatan mood harian untuk melacak pola emosi dan tidur. "
            "Jaga pola makan dan tidur yang konsisten. Hindari alkohol atau zat yang dapat memicu episode mania atau depresi."
        ),
        "terapi": (
            "Bipolar biasanya memerlukan kombinasi pengobatan jangka panjang. Pengobatan dengan mood stabilizer seperti lithium, antipsikotik atipikal, atau antidepresan digunakan sesuai gejala. "
            "Psikoterapi seperti terapi interpersonal atau psychoeducation sangat penting untuk membantu pasien memahami dan mengelola kondisi ini."
        )
    },

    "ptsd": {
        "judul": "Post-Traumatic Stress Disorder (PTSD)",
        "deskripsi": (
            "PTSD adalah reaksi psikologis terhadap peristiwa traumatis, seperti kekerasan, bencana alam, atau kecelakaan berat. "
            "Orang dengan PTSD mengalami kilas balik, mimpi buruk, dan perasaan terancam, bahkan setelah kejadian berakhir. "
            "Kondisi ini tidak hanya memengaruhi pikiran, tetapi juga tubuh dan hubungan sosial."
        ),
        "gejala": (
            "Kilas balik menyakitkan, mimpi buruk berulang, kewaspadaan berlebihan, perasaan mati rasa emosional, "
            "menghindari tempat atau orang yang terkait dengan trauma, perubahan suasana hati yang drastis."
        ),
        "solusi": (
            "Buat rutinitas harian yang terasa aman dan nyaman. Praktikkan grounding techniques saat merasa cemas, seperti menyentuh benda dingin atau menyebutkan lima hal yang terlihat di sekelilingmu."
        ),
        "terapi": (
            "Terapi utama untuk PTSD adalah terapi pemrosesan kognitif (CPT), terapi eksposur, dan EMDR (Eye Movement Desensitization and Reprocessing). "
            "Psikiater bisa memberikan obat seperti SSRI untuk membantu mengurangi gejala kecemasan dan mimpi buruk."
        )
    },
    "adhd": {
    "judul": "Attention Deficit Hyperactivity Disorder (ADHD)",
    "deskripsi": (
        "ADHD adalah gangguan perkembangan saraf yang memengaruhi kemampuan seseorang "
        "untuk memusatkan perhatian, mengendalikan impuls, dan mengatur tingkat aktivitas. "
        "Kondisi ini biasanya muncul sejak masa kanak-kanak, namun bisa berlanjut hingga dewasa."
    ),
    "gejala": (
        "Sulit memusatkan perhatian, sering lupa, mudah teralihkan, kesulitan mengatur tugas, "
        "gelisah atau tidak bisa diam, berbicara berlebihan, impulsif dalam mengambil keputusan."
    ),
    "solusi": (
        "Gunakan pengingat atau to-do list untuk mengatur aktivitas harian. "
        "Bagi tugas besar menjadi bagian-bagian kecil agar lebih mudah diselesaikan. "
        "Istirahat sejenak di antara pekerjaan untuk menjaga fokus."
    ),
    "terapi": (
        "Penanganan ADHD dapat mencakup terapi perilaku, konseling, serta pengobatan seperti "
        "stimulant (methylphenidate) atau non-stimulant sesuai rekomendasi dokter. "
        "Dukungan keluarga dan lingkungan juga berperan besar dalam membantu pengelolaan gejala."
    )
}
}

# === SET BACKGROUND WAVE ===
set_wave_background()

# === SECTION 1 - Hero Section ===
with st.container():
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <h1 style='font-size: 3rem; color: #002F6C; font-weight: bold;'>MentalCare: Deteksi Dini Gangguan Kesehatan Mental</h1>
        <p style="max-width: 550px; font-size: 1.1rem;">
        Aplikasi berbasis AI yang membantu mengenali potensi adanya gangguan kesehatan mental dari media sosial X berdasarkan aktivitas terbaru pengguna dengan mengambil 5 tweet terakhir. Ini bukan alat diagnosis medis, namun bisa menjadi langkah awal untuk memahami kondisi emosional kamu secara lebih dalam. Kamu bisa memilih untuk melakukan deteksi otomatis atau input manual dengan memasukkan username akun X kamu. Kamu bisa memilih untuk otomatis input lewat Username X atau manual input dan ceritakan kondisi mentalmu di story box
        </p>
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <a href="/home" target="_self" class="custom-button">Otomatis</a>
            <a href="/manual_input" target="_self" class="custom-button" style="background-color: #4caf50;">Input Manual</a>
        </div>

        """, unsafe_allow_html=True)
    with col2:
        st.image("assets/images/illustration4.png", width=420)

st.markdown("<br><br>", unsafe_allow_html=True)

# === SECTION 2 - Gangguan Mental ===
with st.container():
    st.markdown("<div class='wave-container'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:2.2rem;'>Jenis Gangguan yang Dikenali Aplikasi</h2>", unsafe_allow_html=True)

    for i, (key, item) in enumerate(gangguan_data.items(), start=1):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"assets/images/pict{i}.png", caption=item['judul'], use_container_width=True)
        with col2:
            st.markdown(f"""
                <h4 style="margin-top: 0;">{item['judul']}</h4>
                <p>{item['deskripsi']}</p>
                <strong>Gejala Umum:</strong>
                <p style="background-color: rgba(255,255,255,0.1); padding: 0.6rem; border-radius: 8px;">{item['gejala']}</p>
                <strong>Saran Penanganan Awal:</strong>
                <p>{item['solusi']}</p>
                <strong>Terapi Medis dan Psikologis:</strong>
                <p>{item['terapi']}</p>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Inisialisasi koneksi DB dan ambil stats
# CSS custom supaya background container lebih full
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        background-color: #3a3f5c;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ambil data statistik dari DB
db = Database()
label_stats = db.get_label_lv2_stats()
db.close()

if label_stats:
    labels = list(label_stats.keys())
    sizes = list(label_stats.values())

    colors = ['#536DFE', '#7986CB', '#64B5F6', '#4FC3F7', '#81D4FA', '#B3E5FC']  
    colors = colors[:len(labels)]

    # Ukuran grafik diperkecil, tanpa facecolor
    fig, ax = plt.subplots(figsize=(5, 3))

    wedges, texts = ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5, edgecolor='w'))

    ax.legend(wedges, labels,
              title="Manifestasi Gangguan",
              title_fontsize=12,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10,
              frameon=False,
              labelspacing=1.2)

    ax.set_title("Distribusi Manifestasi Gangguan Mental", fontsize=14, pad=20)

    # Reset background jadi default putih
    ax.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')

    plt.tight_layout()

    st.pyplot(fig)
else:
    st.write("Belum ada data statistik label gangguan mental.")

# === CTA Contact Section (Full Width) ===
st.markdown("""
<style>
.cta-full {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background-color: #DAEDFF;
    padding: 2rem 1rem;
    border-top: 2px solid #ccc;
    text-align: center;
}
.cta-full a {
    color: #1f3c88;
    font-weight: bold;
    text-decoration: none;
    margin: 0 0.5rem;
}
</style>

<div class="cta-full">
    <h3 style="color: #1f3c88;">Hubungi Kami</h3>
    <p style="color: #333;">Ada pertanyaan, kritik, atau saran? Kami siap mendengar!</p>
    <p style="color: #1f3c88; font-size: 1.1rem; margin-top: 1rem;">
        📞 <a href="tel:+6281234567890">+62 812-3456-7890</a> &nbsp; | &nbsp; 
        📧 <a href="liza:mentalcareapp@gmail.com">mentalcareapp@gmail.com</a>
    </p>
</div>
""", unsafe_allow_html=True)

# === Footer Section (Full Width) ===
st.markdown("""
<style>
.footer-full {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background-color: #DAEDFF;
    padding: 1rem;
    border-top: 2px solid #ccc;
    text-align: center;
}
</style>

<div class="footer-full">
    <p style="margin: 0; font-size: 0.9rem; color: #333;">
        &copy; 2025 MentalCare App — Dibuat dengan hati untuk generasi yang lebih peduli kesehatan mental.
    </p>
</div>
""", unsafe_allow_html=True)

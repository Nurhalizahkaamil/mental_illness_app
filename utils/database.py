import streamlit as st
import psycopg2
import json

class Database:
    def __init__(self):
        """
        Koneksi ke PostgreSQL menggunakan kredensial dari Streamlit secrets.
        """
        self.conn = psycopg2.connect(
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            host=st.secrets["DB_HOST"],
            port=st.secrets.get("DB_PORT", 5432)
        )
        self.cursor = self.conn.cursor()
        self.ensure_schema()

    def ensure_schema(self):
        """
        Pastikan tabel predictions tersedia. Jika tidak, buat baru.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                username TEXT,
                label_lv1 TEXT,
                confidence_lv1 REAL,
                label_lv2 TEXT,
                confidence_lv2 REAL,
                final_status TEXT,
                debug_flag TEXT,
                tweets JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def insert_result(self, result):
        try:
            tweets_text = result["tweets"] if isinstance(result["tweets"], list) else [str(result["tweets"])]
            username = str(result["username"])
            label_lv1 = str(result["label_lv1"])
            confidence_lv1 = float(result["confidence_lv1"].item() if hasattr(result["confidence_lv1"], "item") else result["confidence_lv1"])
            label_lv2 = str(result["label_lv2"])
            confidence_lv2 = float(result["confidence_lv2"].item() if hasattr(result["confidence_lv2"], "item") else result["confidence_lv2"])
            final_status = str(result["final_status"])
            debug_flag = str(result["debug_flag"])

            self.cursor.execute("""
                INSERT INTO predictions (
                    username,
                    label_lv1,
                    confidence_lv1,
                    label_lv2,
                    confidence_lv2,
                    final_status,
                    debug_flag,
                    tweets
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                username,
                label_lv1,
                confidence_lv1,
                label_lv2,
                confidence_lv2,
                final_status,
                debug_flag,
                json.dumps(tweets_text)
            ))
            self.conn.commit()
            print("[INFO] Data berhasil disimpan ke database.")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan data: {e}")
            self.conn.rollback()

    def get_label_lv2_stats(self):
        """
        Ambil jumlah masing-masing label_lv2 yang sudah masuk ke sistem.
        """
        try:
            self.cursor.execute("""
                SELECT label_lv2, COUNT(*) FROM predictions
                GROUP BY label_lv2
                ORDER BY COUNT(*) DESC;
            """)
            results = self.cursor.fetchall()
            return {row[0]: row[1] for row in results}
        except Exception as e:
            print(f"[ERROR] Gagal mengambil statistik label_lv2: {e}")
            return {}

    def close(self):
        """
        Menutup koneksi database dengan aman.
        """
        self.cursor.close()
        self.conn.close()

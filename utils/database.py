import streamlit as st
import psycopg2
import json
import os
from datetime import datetime


class Database:
    def __init__(self):
        """
        Coba koneksi ke PostgreSQL.
        Kalau gagal, fallback ke mode dummy (tanpa DB, simpan ke file lokal).
        """
        self.is_dummy = False
        self.data = []
        self.dummy_file = "dummy_predictions.json"

        # Load file dummy kalau ada
        if os.path.exists(self.dummy_file):
            try:
                with open(self.dummy_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []

        try:
            self.conn = psycopg2.connect(
                dbname=st.secrets["DB_NAME"],
                user=st.secrets["DB_USER"],
                password=st.secrets["DB_PASS"],
                host=st.secrets["DB_HOST"],
                port=st.secrets.get("DB_PORT", 5432)
            )
            self.cursor = self.conn.cursor()
            self.ensure_schema()
            print("[INFO] Terhubung ke PostgreSQL.")
        except Exception as e:
            print(f"[WARNING] Gagal koneksi ke database: {e}")
            print("[INFO] Menggunakan mode dummy (data disimpan di file lokal).")
            self.is_dummy = True

    def ensure_schema(self):
        """
        Buat tabel jika belum ada (mode PostgreSQL saja).
        """
        if self.is_dummy:
            return
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

    def _to_float_safe(self, value):
        """
        Ubah ke float dengan aman (return None kalau gagal).
        """
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None

    def _normalize_tweets(self, tweets):
        """
        Ubah berbagai format tweets menjadi list of string.
        """
        tweets_text = []
        if isinstance(tweets, list):
            for t in tweets:
                if isinstance(t, dict):
                    if "text" in t:
                        tweets_text.append(t["text"])
                    elif "status" in t:
                        tweets_text.append(f"[STATUS:{t['status']}]")
                    else:
                        tweets_text.append(str(t))
                else:
                    tweets_text.append(str(t))
        else:
            tweets_text = [str(tweets)]
        return tweets_text

    def insert_result(self, result):
        """
        Simpan data ke DB atau file dummy sesuai mode.
        """
        try:
            tweets_text = self._normalize_tweets(result.get("tweets", []))
            username = str(result.get("username", "UNKNOWN"))
            label_lv1 = str(result.get("label_lv1") or "UNKNOWN")
            confidence_lv1 = self._to_float_safe(result.get("confidence_lv1"))
            label_lv2 = str(result.get("label_lv2") or "UNKNOWN")
            confidence_lv2 = self._to_float_safe(result.get("confidence_lv2"))
            final_status = str(result.get("final_status") or "UNKNOWN")
            debug_flag = str(result.get("debug_flag") or "")

            if self.is_dummy:
                entry = {
                    "username": username,
                    "label_lv1": label_lv1,
                    "confidence_lv1": confidence_lv1,
                    "label_lv2": label_lv2,
                    "confidence_lv2": confidence_lv2,
                    "final_status": final_status,
                    "debug_flag": debug_flag,
                    "tweets": tweets_text,
                    "created_at": datetime.now().isoformat()
                }
                self.data.append(entry)
                with open(self.dummy_file, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                print("[INFO] Data disimpan (mode dummy).")
            else:
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
                    json.dumps(tweets_text, ensure_ascii=False)
                ))
                self.conn.commit()
                print("[INFO] Data berhasil disimpan ke database.")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan data: {e}")
            if not self.is_dummy:
                self.conn.rollback()

    def get_label_lv2_stats(self):
        """
        Ambil statistik label_lv2.
        """
        try:
            if self.is_dummy:
                stats = {}
                for row in self.data:
                    label = row.get("label_lv2", "UNKNOWN")
                    stats[label] = stats.get(label, 0) + 1
                return stats
            else:
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
        Tutup koneksi kalau bukan mode dummy.
        """
        if not self.is_dummy:
            self.cursor.close()
            self.conn.close()

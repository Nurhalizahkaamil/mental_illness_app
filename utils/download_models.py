import os
import zipfile
import gdown

MODEL_DIR = "models"
ZIP_PATH = "result.zip"
GOOGLE_DRIVE_ID = "1xwdr3BOR3GjcMGAzs-USI_zwlOG7-z6J"  # ID dari link Drive kamu
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

def download_and_extract():
    # Cek apakah direktori model sudah ada dan berisi file
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print("Model folder already exists. Skipping download.")
        return

    # Unduh jika belum ada
    if not os.path.exists(ZIP_PATH):
        print("Downloading result.zip from Google Drive...")
        gdown.download(GOOGLE_DRIVE_URL, ZIP_PATH, quiet=False)
    else:
        print("Zip file already exists. Skipping download.")

    # Ekstrak isi zip ke folder models/
    print("Extracting model files...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print(f"Model extracted to {MODEL_DIR}/")

if __name__ == "__main__":
    download_and_extract()

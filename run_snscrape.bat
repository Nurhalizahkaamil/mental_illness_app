@echo off
setlocal

:: Path ke virtual environment Python 3.10
set VENV_PATH=D:\KULIAH\Skripsi\mental-health-app\.venv310

:: Path ke certifi SSL certificate
set CERT_PATH=%VENV_PATH%\Lib\site-packages\certifi\cacert.pem

:: Aktifkan virtualenv
call %VENV_PATH%\Scripts\activate.bat

:: Set environment variables untuk SSL fix
set SSL_CERT_FILE=%CERT_PATH%
set REQUESTS_CA_BUNDLE=%CERT_PATH%

:: Input username
set /p USERNAME=Masukkan username Twitter (tanpa @): 

:: Jalankan snscrape
snscrape --jsonl --max-results 5 twitter-user %USERNAME%

pause
endlocal

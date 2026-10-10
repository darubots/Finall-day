@echo off
cd /d "%~dp0"
title Smart City, PLTS & AI Vision Dashboard Launcher
echo =========================================================================
echo MEMULAI SMART CITY MONITORING SYSTEM & AI VISION
echo =========================================================================
echo [1/3] Menjalankan Server AI Vision (YOLO CCTV & Smart Waste)...
start "AI Vision Server" python app_vision.py
timeout /t 2 /nobreak >nul

echo [2/3] Menjalankan Web Server Lokal di Port 8000...
echo (Jangan tutup jendela ini selama monitoring berjalan)
echo =========================================================================
start "Dashboard Web Server" python -m http.server 8000
echo Menunggu web server siap...
timeout /t 3 /nobreak >nul

echo [3/3] Membuka Web Dashboard di browser...
start "" http://localhost:8000/website/

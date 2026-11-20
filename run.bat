@echo off
echo ============================================
echo  AirSense - Air Quality Prediction Web App
echo ============================================
echo.
echo Starting Flask server...
echo Open browser: http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.
cd /d %~dp0
call venv\Scripts\activate
python app.py
pause


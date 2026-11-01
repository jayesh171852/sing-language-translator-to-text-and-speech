@echo off
title Sign Language to Text and Speech Conversion
echo Starting Sign Language Recognition System...
python run_app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)


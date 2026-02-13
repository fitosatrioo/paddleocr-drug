@echo off
echo ========================================
echo Installing PaddleOCR CPU Dependencies
echo ========================================
echo.

REM Check if Python is installed
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install requirements
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo You can now run the OCR detector with:
echo   python paddle_ocr_detector.py
echo.
pause

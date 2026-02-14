@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Plastic Waste Detector - Windows Setup
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Check Python version (need 3.10+)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! lss 3 (
    echo [ERROR] Python 3.10+ required. Found: !PYVER!
    pause
    exit /b 1
)
if !PYMAJOR! equ 3 if !PYMINOR! lss 10 (
    echo [ERROR] Python 3.10+ required. Found: !PYVER!
    pause
    exit /b 1
)
echo [OK] Python !PYVER! found.

:: Navigate to script directory
cd /d "%~dp0"
echo [OK] Working directory: %cd%

:: Create virtual environment
if not exist "venv" (
    echo.
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: Activate virtual environment
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

:: Install dependencies
echo.
echo [3/4] Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

:: Download model weights
echo.
echo [4/4] Downloading YOLOv4 model weights...
if not exist "model" mkdir model

set YOLO_WEIGHTS=model\yolov4.weights
set YOLO_TINY_WEIGHTS=model\yolov4-tiny.weights
set YOLO_CFG=model\yolov4.cfg
set YOLO_TINY_CFG=model\yolov4-tiny.cfg
set COCO_NAMES=model\coco.names

set NEED_DOWNLOAD=0
if not exist "!YOLO_WEIGHTS!" set NEED_DOWNLOAD=1
if not exist "!YOLO_TINY_WEIGHTS!" set NEED_DOWNLOAD=1

if !NEED_DOWNLOAD! equ 0 (
    echo [OK] Model weights already downloaded.
    goto :verify
)

python download_models.py --yolov4
if !errorlevel! equ 0 goto :verify

echo.
echo [WARNING] Automatic download failed. Downloading manually...
echo.

if not exist "!YOLO_WEIGHTS!" (
    echo Downloading yolov4.weights ~246 MB ...
    curl -L -o "!YOLO_WEIGHTS!" "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
)
if not exist "!YOLO_TINY_WEIGHTS!" (
    echo Downloading yolov4-tiny.weights ~22 MB ...
    curl -L -o "!YOLO_TINY_WEIGHTS!" "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
)
if not exist "!YOLO_CFG!" (
    echo Downloading yolov4.cfg...
    curl -L -o "!YOLO_CFG!" "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
)
if not exist "!YOLO_TINY_CFG!" (
    echo Downloading yolov4-tiny.cfg...
    curl -L -o "!YOLO_TINY_CFG!" "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
)
if not exist "!COCO_NAMES!" (
    echo Downloading coco.names...
    curl -L -o "!COCO_NAMES!" "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
)

:verify
:: Verify
echo.
echo ============================================
echo   Verifying setup...
echo ============================================
set ALL_GOOD=1

if exist "!YOLO_WEIGHTS!" (
    echo [OK] yolov4.weights
) else (
    echo [MISSING] yolov4.weights
    set ALL_GOOD=0
)
if exist "!YOLO_TINY_WEIGHTS!" (
    echo [OK] yolov4-tiny.weights
) else (
    echo [MISSING] yolov4-tiny.weights
    set ALL_GOOD=0
)
if exist "!YOLO_CFG!" (
    echo [OK] yolov4.cfg
) else (
    echo [MISSING] yolov4.cfg
    set ALL_GOOD=0
)
if exist "!YOLO_TINY_CFG!" (
    echo [OK] yolov4-tiny.cfg
) else (
    echo [MISSING] yolov4-tiny.cfg
    set ALL_GOOD=0
)
if exist "!COCO_NAMES!" (
    echo [OK] coco.names
) else (
    echo [MISSING] coco.names
    set ALL_GOOD=0
)

echo.
if !ALL_GOOD! equ 1 (
    echo ============================================
    echo   Setup complete!
    echo ============================================
    echo.
    echo To start using the detector:
    echo   1. Open a terminal in this folder
    echo   2. Run: venv\Scripts\activate
    echo   3. Run: python detector.py --mode webcam
    echo.
    echo For the backend server:
    echo   python backend.py
    echo.
) else (
    echo ============================================
    echo   Setup incomplete - some files missing
    echo ============================================
    echo   Check the errors above and try again.
    echo.
)

pause
endlocal

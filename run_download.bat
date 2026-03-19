@echo off
REM Download Gemma 3 270M and test it
echo ========================================================================
echo DOWNLOADING GEMMA 3 270M FOR DISTILLATION
echo ========================================================================
echo.

echo Activating bdh environment...
call conda activate bdh

if errorlevel 1 (
    echo ERROR: Failed to activate bdh environment
    echo Please make sure you have conda and bdh environment installed
    pause
    exit /b 1
)

echo.
echo Running download script...
echo.
python download_gemma3_270m.py

echo.
echo ========================================================================
echo Download complete!
echo ========================================================================
pause

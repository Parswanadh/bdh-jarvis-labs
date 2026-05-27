@echo off
echo ============================================================
echo BDH TRAINING MONITOR
echo ============================================================
echo.
echo Activating conda environment...
call conda activate bdh-fest
echo.
echo Starting monitoring...
echo.
cd /d D:\projects\BDH
python monitor_training.py

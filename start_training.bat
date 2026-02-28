@echo off
echo ============================================================
echo BDH REASONING TRAINING - STARTING NOW
echo ============================================================
echo.
echo Activating conda environment...
call conda activate bdh-fest
echo.
echo Starting training...
echo.
cd /d D:\projects\BDH
python train_bdh_reasoning.py
echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
pause

@echo off
echo ============================================================
echo SPRS - Daily Incremental Update (5am Scheduled Task)
echo ============================================================
echo Started at: %date% %time%
echo.

REM Log to file for tracking
set LOG_FILE=C:\Users\user\SPRS_daily_log.txt
echo [%date% %time%] Starting daily update... >> %LOG_FILE%

cd /d C:\Users\user
python nba_daily_update.py

if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Daily update completed successfully >> %LOG_FILE%
) else (
    echo [%date% %time%] Daily update FAILED with error code %ERRORLEVEL% >> %LOG_FILE%
)

echo.
echo Finished at: %date% %time%

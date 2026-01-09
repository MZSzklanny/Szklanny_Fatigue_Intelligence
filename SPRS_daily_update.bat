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
    echo.
    echo Pushing updated data to GitHub...
    echo [%date% %time%] Starting GitHub push... >> %LOG_FILE%

    REM Add updated files to git
    git add NBA_Quarter_ALL_Combined.xlsx
    git add NBA_Training_Full_5Seasons.parquet
    git add player_historical_stats.pkl
    git add player_vs_team_stats.pkl
    git add team_matchup_stats.json
    git add szklanny_streamlit_app.py
    git add player_model_lstm.keras
    git add player_model_scalers.pkl
    git add training_report.txt
    git add nba_daily_update.py

    REM Commit with date stamp
    git commit -m "Daily auto-update: New data for %date%"

    REM Push to GitHub
    git push

    if %ERRORLEVEL% EQU 0 (
        echo [%date% %time%] GitHub push completed successfully >> %LOG_FILE%
    ) else (
        echo [%date% %time%] GitHub push FAILED with error code %ERRORLEVEL% >> %LOG_FILE%
    )
) else (
    echo [%date% %time%] Daily update FAILED with error code %ERRORLEVEL% >> %LOG_FILE%
)

echo.
echo Finished at: %date% %time%

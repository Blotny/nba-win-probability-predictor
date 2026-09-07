@echo off
REM Daily job for Windows Task Scheduler. Schedule it for ~10:00 CET: west-coast
REM games end around 07:00 CET, so yesterday's results are complete in the API
REM and today's games have not started.
REM
REM Note: GAME_DATE from nba_api is the league (ET) date, not the local one.

cd /d "%~dp0"

echo [1/4] closing out played games
venv\Scripts\python.exe src\fetch_data.py --incremental || goto :error

echo [2/4] fetching the schedule
venv\Scripts\python.exe src\fetch_schedule.py || goto :error

echo [3/4] rebuilding features
venv\Scripts\python.exe src\clean_data.py || goto :error
venv\Scripts\python.exe src\feature_engineering.py || goto :error

echo [4/4] scoring
venv\Scripts\python.exe src\predict.py || goto :error
REM Order matters: the FIRST model is primary and is what the dashboard shows.
REM v2 (fit on 2020-26) forecasts - the season ahead is unseen by both, so the
REM model with more and fresher data wins on priors. v1 rides along as a shadow.
REM Pages 1-5 stay on v1 via predict.py, because a backtest needs a holdout.
venv\Scripts\python.exe src\predict_upcoming.py --models models\win_prob_blend_v2_2026-09-07.joblib models\win_prob_blend.joblib || goto :error

echo done
exit /b 0

:error
echo FAILED with code %errorlevel%
exit /b %errorlevel%

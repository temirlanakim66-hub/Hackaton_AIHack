@echo off
echo =====================================================================
echo   SAIGAK-ORTO: 3D Craniometry and Head Deformity Workstation
echo   Local Server URL: http://127.0.0.1:8765
echo =====================================================================
start "" "http://127.0.0.1:8765"
"%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%~dp0server.py"
pause

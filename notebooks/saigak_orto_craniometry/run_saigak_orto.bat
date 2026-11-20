@echo off
chcp 65001 >nul
setlocal
title SAIGAK-ORTO: 3D Craniometry Workstation

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

set "PYTHONHOME="
set "PYTHONPATH="

echo =====================================================================
echo   SAIGAK-ORTO: 3D Craniometry and Head Deformity Workstation
echo   Local Server URL: http://127.0.0.1:8765
echo =====================================================================
echo.

if not exist "%APP_DIR%server.py" goto :missing_project

set "PYTHON_EXE=%APP_DIR%python_runtime\python.exe"
set "PYTHONW_EXE=%APP_DIR%python_runtime\pythonw.exe"

if exist "%PYTHON_EXE%" goto :run_bundled

echo [!] Встроенная папка python_runtime не найдена. Пробуем системный Python...
where python >nul 2>&1
if not errorlevel 1 goto :use_sys_python
where py >nul 2>&1
if not errorlevel 1 goto :use_sys_py

echo.
echo [ОШИБКА] Не найдена папка python_runtime и не установлен Python.
echo Убедитесь, что вы полностью распаковали весь архив с проектом перед запуском.
echo.
pause
exit /b 1

:use_sys_python
set "PYTHON_EXE=python"
goto :run_system

:use_sys_py
set "PYTHON_EXE=py"
goto :run_system

:run_bundled
echo Запуск через встроенный Python (установка не требуется)...
echo Подождите несколько секунд — браузер откроется автоматически.
echo Не закрывайте это окно, пока работаете с программой.
echo.
if not "%SAIGAK_NO_BROWSER%"=="1" if exist "%PYTHONW_EXE%" start "" "%PYTHONW_EXE%" "%APP_DIR%python_runtime\open_browser.py" "http://127.0.0.1:8765"
"%PYTHON_EXE%" "%APP_DIR%server.py"
goto :finished

:run_system
echo Запуск через системный Python...
echo Не закрывайте это окно, пока работаете с программой.
echo.
if not "%SAIGAK_NO_BROWSER%"=="1" start "" "http://127.0.0.1:8765"
"%PYTHON_EXE%" "%APP_DIR%server.py"
goto :finished

:missing_project
echo [ОШИБКА] Рядом с этим файлом не найден server.py!
echo Похоже, вы запустили .bat отдельно или прямо из архива без распаковки.
echo Распакуйте ВСЮ папку saigak_orto_craniometry целиком и запустите снова.
echo.
pause
exit /b 1

:finished
echo.
echo Сервер остановлен.
pause

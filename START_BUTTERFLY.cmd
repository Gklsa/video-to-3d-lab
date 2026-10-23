@echo off
setlocal
pushd "%~dp0"
set "LAB_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "LAB_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0butterfly-mose3\.venv\Scripts\python.exe" set "LAB_PYTHON=%~dp0butterfly-mose3\.venv\Scripts\python.exe"
"%LAB_PYTHON%" -X utf8 -u "%~dp0start_butterfly.py" %*
set "LAB_EXIT=%ERRORLEVEL%"
if not "%LAB_EXIT%"=="0" pause
popd
endlocal & exit /b %LAB_EXIT%

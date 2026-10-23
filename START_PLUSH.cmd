@echo off
setlocal
set "PLUSH_PROJECT=%~dp0plush-phystwin"
set "PLUSH_PYTHON=%PLUSH_PROJECT%\.venv\Scripts\python.exe"
if not exist "%PLUSH_PYTHON%" (
    echo Set up the environment first. See plush-phystwin\README.md.
    pause
    exit /b 1
)
pushd "%PLUSH_PROJECT%"
"%PLUSH_PYTHON%" -X utf8 -u "%PLUSH_PROJECT%\view_plush.py" --browser %*
set "PLUSH_EXIT=%ERRORLEVEL%"
if not "%PLUSH_EXIT%"=="0" pause
popd
endlocal & exit /b %PLUSH_EXIT%

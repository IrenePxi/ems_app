@echo off
echo Starting Daily EMS Sandbox Web Application...
REM Use the venv Python from the parent directory
call ..\..\..venv\Scripts\activate.bat
python app_web.py
pause

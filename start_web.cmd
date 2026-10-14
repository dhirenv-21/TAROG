@echo off
cd /d "%~dp0"
python -m taorg.web --open
if errorlevel 1 pause

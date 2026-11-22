@echo off
echo Installing CogniBot requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Now double-click run_demo.bat
pause

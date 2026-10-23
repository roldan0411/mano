@echo off
rem Abre la aplicacion de escritorio del gemelo de la mano (sin consola)
start "" "%USERPROFILE%\gemelo_mano\venv\Scripts\pythonw.exe" "%~dp0app_gemelo.py"

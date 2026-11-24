@echo off
chcp 65001 >nul
"D:\amancoda\envs\pytorch\python.exe" -B "%~dp0STEP19_六联图可视化诊断.py" --gui
if errorlevel 1 pause

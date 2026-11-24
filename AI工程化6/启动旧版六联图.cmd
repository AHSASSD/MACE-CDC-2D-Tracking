@echo off
chcp 65001 >nul
"D:\amancoda\envs\pytorch\python.exe" -B "%~dp0..\AI工程化4 - 副本\STEP6_事件切片可视化.py" --gui
if errorlevel 1 pause

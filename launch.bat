@echo off
title StarMade Voxelizer
cd /d "%~dp0"
set PYTHONPATH=%~dp0;%PYTHONPATH%
python main.py
pause

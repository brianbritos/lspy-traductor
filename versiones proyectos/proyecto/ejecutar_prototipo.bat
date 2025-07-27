@echo off
chcp 65001 > nul
echo Verificando e instalando dependencias...

REM Usar la ruta correcta del ejecutable de Python
set PYTHON_EXEC="C:\Users\brian\AppData\Local\Microsoft\WindowsApps\python.exe"

REM Verifica si existe
if not exist %PYTHON_EXEC% (
    echo ❌ Python no encontrado en la ruta esperada: %PYTHON_EXEC%
    pause
    exit /b
)

REM Instalar dependencias si es necesario
%PYTHON_EXEC% -m pip install --quiet opencv-python mediapipe joblib numpy

echo Dependencias listas.

echo Ejecutando prediccion en vivo...
%PYTHON_EXEC% src\predict_live.py

echo Programa finalizado.
pause
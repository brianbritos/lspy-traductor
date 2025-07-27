@echo off
echo Verificando instalacion de Python...
where python >nul 2>nul
if errorlevel 1 (
    echo  Python no esta instalado o no se encuentra en el PATH.
    pause
    exit /b
)

echo Instalando dependencias necesarias...
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe pandas

echo Ejecutando captura de senas con movimiento...
python src\collect_data_motion.py

echo Captura finalizada. Presiona cualquier tecla para cerrar.
pause >nul

@echo off
echo Verificando instalacion de Python...
"C:\Python311\python.exe" --version

echo Instalando dependencias necesarias...
"C:\Python311\python.exe" -m pip install --upgrade pip
"C:\Python311\python.exe" -m pip install mediapipe opencv-python pandas

echo Ejecutando captura de señas con movimiento...
"C:\Python311\python.exe" src\collect_data_motion.py

echo Captura finalizada. Presiona cualquier tecla para cerrar.
pause >nul

#!/usr/bin/env python3
"""
gestor_dependencias.py - Gestor offline de dependencias para el proyecto Traductor

Uso:
    python gestor_dependencias.py --download   # Descarga paquetes a ./libs/
    python gestor_dependencias.py --install    # Instala paquetes desde ./libs/
"""

import os
import sys
import subprocess
import argparse

# ------------------- DEPENDENCIAS -------------------
DEPENDENCIAS = [
    "opencv-python>=4.5.5",
    "numpy>=1.22",
    "pandas>=1.3",
    "joblib>=1.1",
    "scikit-learn>=1.0",
    "mediapipe>=0.10",
    "pillow>=9.0",
    "pyttsx3>=2.90"
]

LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")

# ------------------- FUNCIONES -------------------
def run(cmd_list):
    """Ejecuta un comando en la terminal"""
    print(">>", " ".join(cmd_list))
    subprocess.check_call(cmd_list)

def download_packages():
    """Descarga todas las dependencias a la carpeta libs"""
    os.makedirs(LIBS_DIR, exist_ok=True)
    print(f"📥 Descargando dependencias en: {LIBS_DIR}")
    run([sys.executable, "-m", "pip", "download", "-d", LIBS_DIR] + DEPENDENCIAS)
    print("✅ Descarga completada.")

def install_packages():
    """Instala dependencias desde la carpeta libs"""
    if not os.path.exists(LIBS_DIR):
        print("❌ No existe la carpeta 'libs'. Ejecuta primero con --download")
        return
    print(f"📦 Instalando dependencias desde: {LIBS_DIR}")
    run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", LIBS_DIR] + DEPENDENCIAS)
    print("✅ Instalación completada.")

def parse_args():
    parser = argparse.ArgumentParser(description="Gestor offline de dependencias para Traductor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Descargar dependencias a ./libs/")
    group.add_argument("--install", action="store_true", help="Instalar dependencias desde ./libs/")
    return parser.parse_args()

# ------------------- MAIN -------------------
if __name__ == "__main__":
    args = parse_args()
    try:
        if args.download:
            download_packages()
        elif args.install:
            install_packages()
    except subprocess.CalledProcessError as e:
        print("❌ Error ejecutando pip:", e)
        sys.exit(1)

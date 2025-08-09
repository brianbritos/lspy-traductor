import importlib
import subprocess
import sys

# Lista de librerías necesarias
requisitos = [
    "mediapipe",
    "opencv-python",
    "scikit-learn",
    "pandas",
    "numpy"
]

def instalar_paquete(paquete):
    print(f" Instalando {paquete}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

def verificar_requisitos():
    for paquete in requisitos:
        try:
            importlib.import_module(paquete)
            print(f"✅ {paquete} ya está instalado")
        except ImportError:
            instalar_paquete(paquete)

# Verificar al inicio
verificar_requisitos()

# Aquí ya puedes poner el resto del programa con el menú

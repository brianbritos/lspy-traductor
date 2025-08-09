import os
import sys
import subprocess

DEPENDENCIAS = [
    "opencv-python",
    "numpy",
    "pandas",
    "joblib",
    "scikit-learn",
    "mediapipe"
]

def run_cmd(cmd):
    """Ejecuta un comando en la terminal."""
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"Error ejecutando: {cmd}")
        sys.exit(1)

def instalar_dependencias():
    print("="*50)
    print("Instalador de dependencias - Lector de Señas")
    print("="*50)

    # Verificar Python
    print(f"Python detectado: {sys.version.split()[0]}")

    # Actualizar pip
    print("\nActualizando pip...")
    run_cmd(f"{sys.executable} -m pip install --upgrade pip")

    # Instalar cada dependencia
    for dep in DEPENDENCIAS:
        print(f"\nInstalando {dep}...")
        run_cmd(f"{sys.executable} -m pip install {dep}")

    print("\nInstalación completada con éxito.")
    print("Puedes ejecutar tu programa con:")
    print(f"    python {os.path.basename(__file__).replace('setup.py', 'tu_programa.py')}")
    print("="*50)

if __name__ == "__main__":
    instalar_dependencias()

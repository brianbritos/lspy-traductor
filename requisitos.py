#!/usr/bin/env python3
"""
requisitos.py - Instalador robusto de dependencias para el proyecto 'Traductor'

Uso:
    python requisitos.py --venv        # crea .venv e instala allí
    python requisitos.py --system      # instala en el Python activo (global/actual)
    python requisitos.py --write-reqs  # solo escribe requirements.txt y sale
    python requisitos.py --dev         # incluye dependencias de desarrollo (pyinstaller)
    Agregar --upgrade para forzar upgrade de paquetes
    Agregar --check para comprobar imports al final
"""

import sys
import os
import subprocess
import argparse
import venv
import platform

# -------------------------
# Dependencias del proyecto
# -------------------------
CORE_DEPS = [
    "opencv-python>=4.5.5",
    "numpy>=1.22",
    "pandas>=1.3",
    "joblib>=1.1",
    "scikit-learn>=1.0",
    "mediapipe>=0.10",
    "pillow>=9.0",
    "pyttsx3>=2.90"
]
DEV_DEPS = [
    "pyinstaller>=5.8"
]


def run(cmd_list):
    print("Ejecutando:", " ".join(cmd_list))
    subprocess.check_call(cmd_list)


def write_requirements(path="requirements.txt", packages=None):
    packages = packages or CORE_DEPS
    with open(path, "w", encoding="utf-8") as f:
        for p in packages:
            f.write(p + "\n")
    print(f"requirements.txt creado en: {os.path.abspath(path)}")


def create_venv(path=".venv"):
    if os.path.exists(path):
        print(f"El entorno virtual '{path}' ya existe. Se reutilizará.")
    else:
        print(f"Creando entorno virtual en: {path}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(path)
    python_bin = os.path.join(path, "Scripts" if os.name == "nt" else "bin", "python")
    if not os.path.exists(python_bin):
        raise FileNotFoundError(f"No se encontró el ejecutable Python del venv en: {python_bin}")
    return python_bin


def install_with_python(python_exec, packages, upgrade=False):
    cmd = [python_exec, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    run(cmd)


def check_imports_local():
    checks = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "PIL": "pillow",
        "tkinter": "tkinter (paquete del sistema)"
    }
    results = {}
    print("\nVerificando imports en el intérprete actual...")
    for mod, pkg in checks.items():
        try:
            __import__(mod)
            results[pkg] = True
        except Exception:
            results[pkg] = False
    for pkg, ok in results.items():
        print(f" - {pkg:30s}: {'OK' if ok else 'FALTA'}")

    if not results.get("tkinter", True):
        sysname = platform.system()
        print("\nNota: tkinter no está disponible en este intérprete. Instalar según tu OS:")
        if sysname == "Linux":
            print("  Debian/Ubuntu: sudo apt-get install python3-tk")
            print("  Fedora: sudo dnf install python3-tkinter")
        elif sysname == "Darwin":
            print("  macOS: instala Python desde python.org o revisa homebrew (python-tk puede variar).")
        elif sysname == "Windows":
            print("  Windows: tkinter suele venir con el instalador oficial de Python (usa python.org).")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Instalador de dependencias para Traductor (captura, entrenamiento y demo).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--venv", action="store_true", help="Crear .venv e instalar allí")
    group.add_argument("--system", action="store_true", help="Instalar en el Python actual (por defecto si no se especifica --venv)")
    p.add_argument("--write-reqs", action="store_true", help="Solo escribir requirements.txt y salir")
    p.add_argument("--dev", action="store_true", help="Incluir dependencias de desarrollo (pyinstaller)")
    p.add_argument("--upgrade", action="store_true", help="Forzar upgrade (--upgrade pip install)")
    p.add_argument("--check", action="store_true", help="Comprobar imports al finalizar")
    return p.parse_args()


def main():
    args = parse_args()
    packages = CORE_DEPS.copy()
    if args.dev:
        packages += DEV_DEPS

    # Solo escribir requirements.txt
    if args.write_reqs:
        write_requirements(packages=packages)
        if not (args.venv or args.system or args.check):
            return

    # Instalar en venv
    if args.venv:
        py = create_venv(".venv")
        install_with_python(py, packages, upgrade=args.upgrade)
        if args.check:
            run([py, "-c",
                 "import importlib, sys\nmods=['cv2','mediapipe','sklearn','joblib','numpy','pandas','PIL','tkinter']\nfailed=[]\n"
                 "for m in mods:\n  try:\n    importlib.import_module(m)\n  except Exception:\n    failed.append(m)\n"
                 "print('FAILED:'+','.join(failed) if failed else 'ALL_OK')"])
    else:
        # Instalar en el Python actual
        python_exec = sys.executable
        install_with_python(python_exec, packages, upgrade=args.upgrade)
        if args.check:
            check_imports_local()

    print("\n=== Instalación finalizada ===")
    print(" - Para capturar datos / entrenar: python entrenador.py")
    print(" - Para demo en vivo: python traduccion.py")
    print("================================")


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("Error durante instalación:", e)
        sys.exit(1)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

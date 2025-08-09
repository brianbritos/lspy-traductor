# traductor_senas_unificado.py
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import joblib
import collections

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import mediapipe as mp

# ----------------------------
# RUTAS SEGURAS (funciona .py y .exe)
# ----------------------------
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(_file_))

BASE_PATH = get_base_path()
DATA_DIR = os.path.join(BASE_PATH, "data")         # aquí guardamos CSV y subcarpetas dinámicas
MODEL_PATH = os.path.join(BASE_PATH, "model.pkl")  # donde se guardará el modelo

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Configuración MediaPipe / utilidades
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

EXPECTED_FEATURES = 63  # 21 landmarks * 3 (x,y,z)
COLUMN_NAMES = [f"kp_{i}" for i in range(EXPECTED_FEATURES)] + ["label"]

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return None

# ----------------------------
# FUNCIONES: captura estática
# ----------------------------
def capturar_estaticas():
    try:
        NUM_SAMPLES = int(input("¿Cuántas muestras estáticas querés capturar? (ej: 50): ").strip())
    except ValueError:
        print("Número inválido. Cancelando.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    OUTPUT_DIR = DATA_DIR  # guardamos un CSV por etiqueta en data/
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label = input("Ingresá el nombre de la seña estática (label): ").strip().lower()
    output_path = os.path.join(OUTPUT_DIR, f"{label}.csv")

    samples = []
    print("\nInstrucciones:")
    print(" - Colocate frente a la cámara.")
    print(" - Cuando la mano esté en la posición correcta, presiona 'c' para iniciar una verificación de estabilidad.")
    print(" - El sistema comprobará 10 frames en ráfaga y, si son estables, se guardará la muestra.")
    print(" - Presioná ESC para cancelar.\n")

    muestras_guardadas = 0
    cv2.namedWindow("Captura Estática", cv2.WINDOW_NORMAL)

    while muestras_guardadas < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Mostrar landmarks si detecta
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Guardadas: {muestras_guardadas}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Presiona 'c' para verificar estabilidad y capturar", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Captura Estática", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Recopilar 10 frames consecutivos y comprobar estabilidad
            burst = []
            ok = True
            for _ in range(10):
                ret_b, f_b = cap.read()
                if not ret_b:
                    ok = False
                    break
                f_b = cv2.flip(f_b, 1)
                r_b = hands.process(cv2.cvtColor(f_b, cv2.COLOR_BGR2RGB))
                if not r_b.multi_hand_landmarks:
                    ok = False
                    break
                lms = r_b.multi_hand_landmarks[0].landmark
                if len(lms) != 21:
                    ok = False
                    break
                kp = [c for lm in lms for c in (lm.x, lm.y, lm.z)]
                burst.append(kp)
                cv2.imshow("Verificando estabilidad", f_b)
                cv2.waitKey(30)

            if not ok or len(burst) < 10:
                print("Mano inestable o puntos incompletos. Repetí.")
                continue

            # comprobación de estabilidad (varianza pequeña)
            arr = np.array(burst)
            var_mean = np.var(arr, axis=0).mean()
            if var_mean > 0.0015:   # umbral ajustable
                print("⚠ Movimiento detectado (no estable). Repetí la captura.")
                continue

            sample = np.mean(arr, axis=0).tolist()
            samples.append(sample)
            muestras_guardadas += 1
            print(f"Captura válida guardada ({muestras_guardadas}/{NUM_SAMPLES}).")

        elif key == 27:  # ESC
            print("Cancelado por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        df = pd.DataFrame(samples)
        df['label'] = label
        # Si existe previamente, concatenar
        if os.path.exists(output_path):
            try:
                prev = pd.read_csv(output_path)
                df = pd.concat([prev, df], ignore_index=True)
            except Exception as e:
                print(f"⚠ No se pudo concatenar con archivo previo: {e}")
        df.to_csv(output_path, index=False)
        print(f"Guardado {len(samples)} muestras en: {output_path}")
    else:
        print("No se guardaron muestras.")

# ----------------------------
# FUNCIONES: captura dinámica (secuencias)
# ----------------------------
def capturar_dinamicas():
    try:
        NUM_SAMPLES = int(input("¿Cuántas secuencias dinámicas querés capturar? (ej: 20): ").strip())
    except ValueError:
        print("Número inválido. Cancelando.")
        return

    try:
        DURATION = float(input("Duración (segundos) por secuencia (ej: 1.0): ").strip())
    except ValueError:
        print("Duración inválida. Cancelando.")
        return

    label = input("Ingresá el nombre de la seña dinámica (label): ").strip().lower()
    OUTPUT_DIR = os.path.join(DATA_DIR, label)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    muestras_guardadas = 0
    print("\nInstrucciones:")
    print(" - Presiona 'c' para iniciar la grabación de la secuencia.")
    print(" - Durante la grabación, mantené la mano visible y en cuadro.")
    print(" - Presioná ESC para cancelar.\n")

    cv2.namedWindow("Captura Dinámica", cv2.WINDOW_NORMAL)
    frames_per_seq = max(1, int(DURATION * 30))  # asumiendo ~30fps

    while muestras_guardadas < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Guardadas: {muestras_guardadas}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, "Presiona 'c' para grabar la secuencia", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Captura Dinámica", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            seq = []
            validos = 0
            total = 0
            start_time = time.time()
            while total < frames_per_seq:
                ret2, f2 = cap.read()
                if not ret2:
                    break
                f2 = cv2.flip(f2, 1)
                res = hands.process(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                total += 1

                if res.multi_hand_landmarks and len(res.multi_hand_landmarks[0].landmark) == 21:
                    kp = [c for lm in res.multi_hand_landmarks[0].landmark for c in (lm.x, lm.y, lm.z)]
                    seq.append(kp)
                    validos += 1
                    mp_drawing.draw_landmarks(f2, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                cv2.putText(f2, f"Grabando secuencia... {total}/{frames_per_seq}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Captura Dinámica", f2)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Validación: al menos 80% frames válidos
            ratio = (validos / total) if total > 0 else 0
            if ratio >= 0.8 and len(seq) > 0:
                filename = os.path.join(OUTPUT_DIR, f"sample_{muestras_guardadas+1}.csv")
                pd.DataFrame(seq).to_csv(filename, index=False)
                muestras_guardadas += 1
                print(f"Secuencia guardada: {filename} ({validos}/{total} válidos, {ratio:.2f})")
            else:
                print(f"Secuencia descartada ({validos}/{total} válidos, {ratio:.2f}). Reintentá.")

        elif key == 27:
            print("Cancelado por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Captura dinámica finalizada.")

# ----------------------------
# FUNCIONES: cargar datos estáticos con confirmación
# ----------------------------
def cargar_datos_estaticos():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and os.path.isfile(os.path.join(DATA_DIR, f))])
    if not files:
        print("No se encontraron archivos .csv estáticos en la carpeta 'data/'.")
        return None

    print("Archivos estáticos disponibles:")
    for i, f in enumerate(files, 1):
        print(f" {i}. {f}")

    try:
        max_load = int(input(f"¿Cuántos de estos archivos querés cargar? (máx {len(files)}): ").strip())
    except ValueError:
        print("Entrada inválida.")
        return None

    if max_load <= 0:
        print("Cantidad inválida.")
        return None

    chosen = files[:min(max_load, len(files))]
    print("Vas a cargar estos archivos:")
    for f in chosen:
        print(" -", f)
    confirm = input("Confirmás la carga? (s/n): ").strip().lower()
    if confirm != 's':
        print("Carga cancelada.")
        return None

    all_df = []
    for f in chosen:
        p = os.path.join(DATA_DIR, f)
        df = read_csv_safe(p)
        if df is None:
            continue
        # Validar formato: debe tener 'label' y 63 features
        if 'label' in df.columns and df.shape[1] == EXPECTED_FEATURES + 1:
            df.columns = COLUMN_NAMES
            all_df.append(df)
        else:
            print(f"⚠ Ignorado {f}: formato no coincide (esperado: 63 cols + 'label').")

    if not all_df:
        print("No se cargó ningún archivo válido.")
        return None

    combined = pd.concat(all_df, ignore_index=True)
    print(f"Cargados {len(combined)} muestras desde archivos estáticos seleccionados.")
    return combined

# ----------------------------
# FUNCIONES: entrenar modelo (une estático y dinámico)
# ----------------------------
def entrenar_modelo():
    print("Iniciando proceso de preparación de datos para entrenamiento...")

    # 1) Cargar estáticos si el usuario quiere
    use_stat = input("¿Querés cargar archivos estáticos ahora? (s/n): ").strip().lower()
    static_df = pd.DataFrame()
    if use_stat == 's':
        static_df = cargar_datos_estaticos() or pd.DataFrame()

    # 2) Buscar carpetas dinámicas (subcarpetas en data/)
    dynamic_rows = []
    for entry in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, entry)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    p = os.path.join(folder, f)
                    df = read_csv_safe(p)
                    if df is None:
                        continue
                    if df.shape[1] == EXPECTED_FEATURES:
                        descriptor = df.mean().tolist()
                        descriptor.append(entry)  # label = carpeta
                        dynamic_rows.append(descriptor)
                    else:
                        print(f"Ignorado {p}: columnas={df.shape[1]} (se esperan {EXPECTED_FEATURES})")

    dynamic_df = pd.DataFrame(dynamic_rows, columns=COLUMN_NAMES) if dynamic_rows else pd.DataFrame()

    # 3) Combinar
    pieces = []
    if not static_df.empty:
        pieces.append(static_df)
    if not dynamic_df.empty:
        pieces.append(dynamic_df)

    if not pieces:
        print("No hay datos para entrenar. Asegurate de tener datos estáticos o dinámicos.")
        return

    dataset = pd.concat(pieces, ignore_index=True)

    # Validación final
    if dataset.shape[1] != EXPECTED_FEATURES + 1:
        print(f"Dataset combinado tiene {dataset.shape[1]} columnas. Se esperan {EXPECTED_FEATURES+1}.")
        return

    X = dataset.drop('label', axis=1)
    y = dataset['label']

    print("Entrenando modelo RandomForest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Resultados en conjunto de test ===")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

# ----------------------------
# FUNCIONES: predicción en vivo
# ----------------------------
def predecir_en_vivo():
    if not os.path.exists(MODEL_PATH):
        print("No se encontró el modelo. Entrená primero.")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"⚠ Error cargando el modelo: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    frame_buffer = collections.deque(maxlen=15)
    print("Mostrá una seña a la cámara. Presioná ESC para salir.")

    cv2.namedWindow("Predicción en Vivo", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks and len(res.multi_hand_landmarks[0].landmark) == 21:
            kp = [c for lm in res.multi_hand_landmarks[0].landmark for c in (lm.x, lm.y, lm.z)]
            frame_buffer.append(kp)

            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        prediction = "..."
        color = (150,150,150)
        if len(frame_buffer) >= 10:
            arr = np.mean(frame_buffer, axis=0).reshape(1, -1)
            try:
                prediction = model.predict(arr)[0]
                color = (0,255,0)
            except Exception as e:
                print(f"⚠ Error en predicción: {e}")

        cv2.putText(frame, f"Seña: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Predicción en Vivo", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# MENÚ PRINCIPAL
# ----------------------------
def menu():
    while True:
        print("\n=== TRADUCTOR DE SEÑAS (LSPy) ===")
        print("1) Capturar señas estáticas")
        print("2) Capturar señas dinámicas (movimiento)")
        print("3) Cargar datos estáticos (selección y confirmación)")
        print("4) Entrenar modelo (usa estáticos y dinámicos)")
        print("5) Predicción en vivo")
        print("6) Salir")
        choice = input("Elegí opción: ").strip()

        if choice == '1':
            capturar_estaticas()
        elif choice == '2':
            capturar_dinamicas()
        elif choice == '3':
            _ = cargar_datos_estaticos()
        elif choice == '4':
            entrenar_modelo()
        elif choice == '5':
            predecir_en_vivo()
        elif choice == '6':
            print("Saliendo. ¡Éxitos con el proyecto!")
            break
        else:
            print("Opción inválida.")

if _name_ == "_main_":
    menu()
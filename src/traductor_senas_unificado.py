import importlib
import subprocess
import sys
import os
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
# Función para instalar librerías si no están
# ----------------------------
def instalar_paquete(paquete):
    print(f"Verificando / instalando paquete: {paquete} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

def verificar_requisitos():
    paquetes = [
        "numpy",
        "pandas",
        "scikit-learn",
        "mediapipe",
        "opencv-python",
        "joblib"
    ]
    for paquete in paquetes:
        try:
            importlib.import_module(paquete)
            print(f"{paquete} ya está instalado.")
        except ImportError:
            instalar_paquete(paquete)

# Verificamos al iniciar el programa
verificar_requisitos()

# ----------------------------
# RUTAS SEGURAS (funciona .py y .exe)
# ----------------------------
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
DATA_DIR = os.path.join(BASE_PATH, "data")         # carpeta para CSVs
MODEL_PATH = os.path.join(BASE_PATH, "model.pkl")  # modelo guardado

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
# Función para elegir mano más centrada y cercana
# ----------------------------
def seleccionar_mano_prioritaria(manos_detectadas, frame_shape=None):
    best_hand = None
    min_dist_score = float('inf')
    for hand in manos_detectadas:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        center_x, center_y = np.mean(xs), np.mean(ys)
        z_mean = np.mean([lm.z for lm in hand.landmark])
        dist_score = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 + abs(z_mean)
        if dist_score < min_dist_score:
            min_dist_score = dist_score
            best_hand = hand
    return best_hand

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

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
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

        hand_landmarks = None
        min_dist_score = float('inf')
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                center_x, center_y = np.mean(xs), np.mean(ys)
                z_mean = np.mean([lm.z for lm in hand.landmark])
                dist_score = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 + abs(z_mean)
                if dist_score < min_dist_score:
                    min_dist_score = dist_score
                    hand_landmarks = hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Guardadas: {muestras_guardadas}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Presiona 'c' para verificar estabilidad y capturar", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Captura Estática", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
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
                hand_b = None
                min_dist_b = float('inf')
                for hand in r_b.multi_hand_landmarks:
                    xs = [lm.x for lm in hand.landmark]
                    ys = [lm.y for lm in hand.landmark]
                    center_x, center_y = np.mean(xs), np.mean(ys)
                    z_mean = np.mean([lm.z for lm in hand.landmark])
                    dist_score = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 + abs(z_mean)
                    if dist_score < min_dist_b:
                        min_dist_b = dist_score
                        hand_b = hand
                if hand_b is None or len(hand_b.landmark) != 21:
                    ok = False
                    break
                kp = [c for lm in hand_b.landmark for c in (lm.x, lm.y, lm.z)]
                burst.append(kp)
                cv2.imshow("Verificando estabilidad", f_b)
                cv2.waitKey(30)

            if not ok or len(burst) < 10:
                print("Mano inestable o puntos incompletos. Repetí.")
                continue

            arr = np.array(burst)
            var_mean = np.var(arr, axis=0).mean()
            if var_mean > 0.0015:
                print("⚠ Movimiento detectado (no estable). Repetí la captura.")
                continue

            sample = np.mean(arr, axis=0).tolist()
            samples.append(sample)
            muestras_guardadas += 1
            print(f"Captura válida guardada ({muestras_guardadas}/{NUM_SAMPLES}).")

        elif key == 27:
            print("Cancelado por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        df = pd.DataFrame(samples)
        df.columns = COLUMN_NAMES[:-1]  # sólo features
        df['label'] = label
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

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    muestras_guardadas = 0
    print("\nInstrucciones:")
    print(" - Presiona 'c' para iniciar la grabación de la secuencia.")
    print(" - Durante la grabación, mantené la mano visible y en cuadro.")
    print(" - Presioná ESC para cancelar.\n")

    cv2.namedWindow("Captura Dinámica", cv2.WINDOW_NORMAL)
    frames_per_seq = max(1, int(DURATION * 30))

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

                hand_b = None
                min_dist_b = float('inf')
                if res.multi_hand_landmarks:
                    for hand in res.multi_hand_landmarks:
                        xs = [lm.x for lm in hand.landmark]
                        ys = [lm.y for lm in hand.landmark]
                        center_x, center_y = np.mean(xs), np.mean(ys)
                        z_mean = np.mean([lm.z for lm in hand.landmark])
                        dist_score = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 + abs(z_mean)
                        if dist_score < min_dist_b:
                            min_dist_b = dist_score
                            hand_b = hand
                if hand_b and len(hand_b.landmark) == 21:
                    kp = [c for lm in hand_b.landmark for c in (lm.x, lm.y, lm.z)]
                    seq.append(kp)
                    validos += 1
                    mp_drawing.draw_landmarks(f2, hand_b, mp_hands.HAND_CONNECTIONS)

                cv2.putText(f2, f"Grabando secuencia... {total}/{frames_per_seq}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Captura Dinámica", f2)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Cancelado por el usuario.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

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
# Cargar datos estáticos con confirmación
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
# Entrenar modelo (une estático y dinámico)
# ----------------------------
def entrenar_modelo():
    print("Iniciando proceso de preparación de datos para entrenamiento...")

    use_stat = input("¿Querés cargar archivos estáticos ahora? (s/n): ").strip().lower()
    static_df = pd.DataFrame()
    if use_stat == 's':
        static_df = cargar_datos_estaticos() or pd.DataFrame()

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

    pieces = []
    if not static_df.empty:
        pieces.append(static_df)
    if not dynamic_df.empty:
        pieces.append(dynamic_df)

    if not pieces:
        print("No hay datos para entrenar. Asegurate de tener datos estáticos o dinámicos.")
        return

    dataset = pd.concat(pieces, ignore_index=True)

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
# Predicción en vivo
# ----------------------------
def predecir_en_vivo():
    if not os.path.exists(MODEL_PATH):
        print("No se encontró el modelo. Entrená primero.")
        return

    model = joblib.load(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    print("Iniciando predicción en vivo. Presiona ESC para salir.")
    cv2.namedWindow("Predicción en vivo", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            best_hand = seleccionar_mano_prioritaria(res.multi_hand_landmarks, frame.shape)
            if best_hand:
                kp = [c for lm in best_hand.landmark for c in (lm.x, lm.y, lm.z)]
                if len(kp) == EXPECTED_FEATURES:
                    pred = model.predict([kp])[0]
                    pred_prob = max(model.predict_proba([kp])[0])
                    text = f"Predicción: {pred} ({pred_prob*100:.1f}%)"
                    mp_drawing.draw_landmarks(frame, best_hand, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Predicción en vivo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Menú principal
# ----------------------------
def menu():
    while True:
        print("\n--- Menú ---")
        print("1. Capturar muestras estáticas")
        print("2. Capturar muestras dinámicas")
        print("3. Entrenar modelo")
        print("4. Predicción en vivo")
        print("5. Salir")

        opcion = input("Elegí una opción (1-5): ").strip()
        if opcion == '1':
            capturar_estaticas()
        elif opcion == '2':
            capturar_dinamicas()
        elif opcion == '3':
            entrenar_modelo()
        elif opcion == '4':
            predecir_en_vivo()
        elif opcion == '5':
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intentá de nuevo.")

if __name__ == "__main__":
    menu()

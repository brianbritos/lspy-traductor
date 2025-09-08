#!/usr/bin/env python3
"""
entrenador.py - Captura de datos y entrenamiento del modelo de señas
Uso:
    python entrenador.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp

# ---------------------------- PATHS ----------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "modelo.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------- CONFIGURACIÓN ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
EXPECTED_FEATURES = 63
COLUMN_NAMES = [f"kp_{i}" for i in range(EXPECTED_FEATURES)] + ["label"]

# ---------------------------- UTILIDADES ----------------------------
def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return None

def get_closest_hand(landmarks_list):
    """Selecciona la mano más cercana al centro de la cámara"""
    if not landmarks_list:
        return None
    min_dist_score = float('inf')
    best_hand = None
    for hand in landmarks_list:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        z_mean = np.mean([lm.z for lm in hand.landmark])
        dist_score = ((np.mean(xs)-0.5)**2 + (np.mean(ys)-0.5)**2)**0.5 + abs(z_mean)
        if dist_score < min_dist_score:
            min_dist_score = dist_score
            best_hand = hand
    return best_hand

def extract_features_np(hand_landmarks):
    """Convierte landmarks en un vector NumPy de 63 valores (x,y,z)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32).ravel()

# ---------------------------- CAPTURA ESTÁTICA ----------------------------
def capturar_estaticas():
    try:
        num_samples = int(input("Número de muestras estáticas a capturar: "))
    except:
        print("Número inválido.")
        return
    label = input("Etiqueta (label) para estas muestras: ").strip().lower()
    output_path = os.path.join(DATA_DIR, f"{label}.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)
    samples, guardadas = [], 0

    print("Presiona 'c' para capturar, 'ESC' para salir.")
    while guardadas < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_landmarks = get_closest_hand(results.multi_hand_landmarks) if results.multi_hand_landmarks else None
        if hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Guardadas: {guardadas}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Captura Estática", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and hand_landmarks:
            # Guardar promedio de 10 frames estables
            burst = []
            for _ in range(10):
                ret_b, f_b = cap.read()
                if not ret_b:
                    break
                f_b = cv2.flip(f_b, 1)
                r_b = hands.process(cv2.cvtColor(f_b, cv2.COLOR_BGR2RGB))
                hand_b = get_closest_hand(r_b.multi_hand_landmarks) if r_b.multi_hand_landmarks else None
                if hand_b is None or len(hand_b.landmark) != 21:
                    break
                kp = extract_features_np(hand_b)
                burst.append(kp)
            if len(burst) == 10 and np.var(np.array(burst), axis=0).mean() <= 0.0015:
                samples.append(np.mean(np.array(burst), axis=0).tolist())
                guardadas += 1
                print(f"Muestra {guardadas}/{num_samples} guardada.")
            else:
                print("Mano inestable, intenta de nuevo.")
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        df = pd.DataFrame(samples, columns=[f"kp_{i}" for i in range(EXPECTED_FEATURES)])
        df['label'] = label
        if os.path.exists(output_path):
            prev = read_csv_safe(output_path)
            if prev is not None and set(prev.columns) == set(df.columns):
                df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(output_path, index=False)
        print(f"{len(samples)} muestras guardadas en {output_path}")

# ---------------------------- CAPTURA DINÁMICA ----------------------------
def capturar_dinamicas():
    try:
        num_samples = int(input("Número de secuencias dinámicas a capturar: "))
        duration = float(input("Duración por secuencia (s): "))
    except:
        print("Entrada inválida.")
        return
    label = input("Etiqueta (label) para estas secuencias: ").strip().lower()
    output_dir = os.path.join(DATA_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)
    guardadas, frames_per_seq = 0, max(1, int(duration * 30))

    print("Presiona 'c' para grabar, 'ESC' para salir.")
    while guardadas < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Guardadas: {guardadas}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Captura Dinámica", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            seq, validos, total = [], 0, 0
            while total < frames_per_seq:
                ret2, f2 = cap.read()
                if not ret2:
                    break
                f2 = cv2.flip(f2, 1)
                res = hands.process(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                total += 1
                hand_b = get_closest_hand(res.multi_hand_landmarks) if res.multi_hand_landmarks else None
                if hand_b and len(hand_b.landmark) == 21:
                    kp = extract_features_np(hand_b)
                    seq.append(kp.tolist())
                    validos += 1
                    mp_drawing.draw_landmarks(f2, hand_b, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Captura Dinámica", f2)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            if validos / total >= 0.8 and seq:
                filename = os.path.join(output_dir, f"sample_{guardadas+1}.csv")
                pd.DataFrame(seq).to_csv(filename, index=False)
                guardadas += 1
                print(f"Secuencia guardada: {filename}")
            else:
                print("Secuencia descartada (demasiados frames inválidos).")
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Captura dinámica finalizada.")

# ---------------------------- ENTRENAMIENTO ----------------------------
def cargar_estaticos():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    dfs = []
    for f in files:
        df = read_csv_safe(os.path.join(DATA_DIR, f))
        if df is not None and df.shape[1] == EXPECTED_FEATURES + 1:
            df.columns = COLUMN_NAMES
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def entrenar_modelo():
    static_df = cargar_estaticos()
    dynamic_rows = []
    for entry in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, entry)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    df = read_csv_safe(os.path.join(folder, f))
                    if df is not None and df.shape[1] == EXPECTED_FEATURES:
                        descriptor = df.mean().tolist()
                        descriptor.append(entry)
                        dynamic_rows.append(descriptor)
    dynamic_df = pd.DataFrame(dynamic_rows, columns=COLUMN_NAMES) if dynamic_rows else pd.DataFrame()
    dataset = pd.concat([static_df, dynamic_df], ignore_index=True) if not static_df.empty or not dynamic_df.empty else None
    if dataset is None or dataset.empty:
        print("No hay datos para entrenar.")
        return

    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n--- Resultados de validación ---")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

# ---------------------------- MENÚ ----------------------------
def menu():
    while True:
        print("\n=== Menú Entrenador ===")
        print("1. Capturar muestras estáticas")
        print("2. Capturar secuencias dinámicas")
        print("3. Entrenar modelo")
        print("0. Salir")
        choice = input("Opción: ").strip()
        if choice == '1':
            capturar_estaticas()
        elif choice == '2':
            capturar_dinamicas()
        elif choice == '3':
            entrenar_modelo()
        elif choice == '0':
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu()

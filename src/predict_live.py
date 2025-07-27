# === predict_live.py (mejorado para manejar señas estáticas y dinámicas) ===
import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import collections

MODEL_PATH = r"C:\Users\brian\Documents\proyecto\model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"No se encontró el modelo en: {MODEL_PATH}")
    input("Presioná una tecla para salir...")
    exit()

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print("Mostrá una seña a la cámara. Presioná ESC para salir.")

# Almacenamiento de historial para predicción de movimiento
frame_buffer = collections.deque(maxlen=15)  # ~0.5 segundos a 30fps

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        if len(hand_landmarks.landmark) == 21:
            keypoints = [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)]
            frame_buffer.append(keypoints)

    prediction = "..."
    color = (100, 100, 100)

    if len(frame_buffer) >= 10:
        # Seña dinámica: promedio de los últimos frames
        averaged = np.mean(frame_buffer, axis=0).reshape(1, -1)
        try:
            prediction = model.predict(averaged)[0]
            color = (0, 255, 0)
        except Exception as e:
            print(f"⚠️ Error en predicción: {e}")

    cv2.putText(frame, f"Seña: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Predicción en Vivo", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

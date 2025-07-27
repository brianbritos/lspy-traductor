#predict_live.py
import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

MODEL_PATH = r"C:\Users\brian\Documents\proyecto\model.pkl"

# Verificar que el modelo existe
if not os.path.exists(MODEL_PATH):
    print(f"No se encontró el modelo en: {MODEL_PATH}")
    input("Presiona una tecla para salir...")
    exit()

# Cargar modelo
model = joblib.load(MODEL_PATH)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print("Mostrá una seña a la cámara. Presioná ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        if len(hand_landmarks.landmark) == 21:
            try:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                # Validación geométrica segura: índice y pulgar
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                distance = np.linalg.norm([
                    index_tip.x - thumb_tip.x,
                    index_tip.y - thumb_tip.y,
                    index_tip.z - thumb_tip.z
                ])

                # Solo predecir si hay separación significativa
                if distance > 0.05:
                    prediction = model.predict([keypoints])[0]
                    cv2.putText(frame, f"Seña: {prediction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"⚠️ Error en validación: {e}")
        else:
            print("⚠️ Mano detectada, pero sin los 21 puntos.")
    else:
        # Nada detectado, no hay predicción
        pass

    cv2.imshow("Predicción en Vivo", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

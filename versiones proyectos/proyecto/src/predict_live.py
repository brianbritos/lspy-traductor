import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

MODEL_PATH = r"C:\Users\brian\Documents\proyecto\model.pkl"
CONFIDENCE_THRESHOLD = 0.7  # ajusta según prueba

if not os.path.exists(MODEL_PATH):
    print(f"No se encontró el modelo en: {MODEL_PATH}")
    input("Presiona una tecla para salir...")
    exit()

model = joblib.load(MODEL_PATH)

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
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

        proba = model.predict_proba([keypoints])[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction = model.classes_[pred_idx]
            text = f"Seña: {prediction} ({confidence:.2f})"
        else:
            text = "Seña: Nada claro"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No se detecta mano", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Predicción en Vivo", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pandas as pd
import os

# Solicitar nombre de la seña al usuario
LABEL = input("Ingresá el nombre de la sena: ").strip().lower()
OUTPUT_PATH = f"data/{LABEL}.csv"
NUM_SAMPLES = 50  # Cuántas veces vas a repetir la seña

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
samples = []

print(f"Muestra la sena '{LABEL}' {NUM_SAMPLES} veces. Presioná ESPACIO cada vez que esté lista.")

while len(samples) < NUM_SAMPLES:
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

        cv2.putText(frame, f"Muestra {len(samples)+1}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            samples.append(keypoints)
            print(f"✅ Capturado {len(samples)}")

    cv2.imshow("Captura de Senas", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Guardar datos
os.makedirs('data', exist_ok=True)
df = pd.DataFrame(samples)
df['label'] = LABEL
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Datos guardados en {os.path.abspath(OUTPUT_PATH)}")

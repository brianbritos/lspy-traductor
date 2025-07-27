import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

LABEL = input("Ingresá el nombre de la seña estática: ").strip().lower()
OUTPUT_PATH = f"data/{LABEL}.csv"
NUM_SAMPLES = 50

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
samples = []

print(f"Muestra la seña '{LABEL}' {NUM_SAMPLES} veces. Presioná ESPACIO cuando estés listo.")

def es_estable(frames, umbral=0.001):
    return np.var(frames, axis=0).mean() < umbral

while len(samples) < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        if len(hand_landmarks.landmark) == 21:
            frames = []
            for _ in range(10):
                ret, f = cap.read()
                if not ret:
                    continue
                f = cv2.flip(f, 1)
                r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                if r.multi_hand_landmarks:
                    lms = r.multi_hand_landmarks[0].landmark
                    if len(lms) == 21:
                        kp = [c for lm in lms for c in (lm.x, lm.y, lm.z)]
                        frames.append(kp)
                    else:
                        frames = []
                        break
                else:
                    frames = []
                    break
                cv2.imshow("Verificando estabilidad", f)
                if cv2.waitKey(30) & 0xFF == 27:
                    break

            if len(frames) == 10 and es_estable(frames):
                sample = np.mean(frames, axis=0).tolist()
                samples.append(sample)
                print(f"✅ Captura {len(samples)}/{NUM_SAMPLES} confirmada.")
            else:
                print("⚠️ Mano inestable o puntos incompletos.")

    cv2.putText(frame, f"Muestras: {len(samples)}/{NUM_SAMPLES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Captura de Seña Estática", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

os.makedirs('data', exist_ok=True)
df = pd.DataFrame(samples)
df['label'] = LABEL
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Datos guardados en {os.path.abspath(OUTPUT_PATH)}")

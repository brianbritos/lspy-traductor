import cv2
import mediapipe as mp
import pandas as pd
import os
import time

LABEL = input("Ingresá el nombre de la seña con movimiento: ").strip().lower()
DURATION = int(input("Duración (segundos) de cada muestra: "))
NUM_SAMPLES = int(input("Cantidad de muestras: "))

OUTPUT_DIR = f"data/{LABEL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print(f"Se capturarán {NUM_SAMPLES} muestras de '{LABEL}' ({DURATION}s cada una). Presioná ENTER para comenzar cada muestra.")

for i in range(1, NUM_SAMPLES + 1):
    input(f"\nPrepararse para muestra {i}/{NUM_SAMPLES}...")
    frames = []
    total_frames = 0
    validos = 0
    start = time.time()

    while time.time() - start < DURATION:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        total_frames += 1

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if len(hand_landmarks.landmark) == 21:
                kp = [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)]
                frames.append(kp)
                validos += 1

        cv2.putText(frame, f"Grabando muestra {i}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Captura Movimiento", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if validos / total_frames >= 0.8:
        pd.DataFrame(frames).to_csv(os.path.join(OUTPUT_DIR, f"sample_{i}.csv"), index=False)
        print(f" Muestra {i} guardada con {validos}/{total_frames} frames válidos.")
    else:
        print(f" Muestra {i} descartada por baja calidad: {validos}/{total_frames} válidos.")

cap.release()
cv2.destroyAllWindows()
print(f"\n Captura de señas '{LABEL}' finalizada.")

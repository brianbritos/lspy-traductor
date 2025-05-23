import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Solicitar nombre de la seña y duración
LABEL = input("Ingresá el nombre de la seña con movimiento: ").strip().lower()
DURATION = int(input("Duración en segundos de cada muestra: "))
NUM_SAMPLES = int(input("Cantidad de muestras a capturar: "))

# Ruta de guardado
OUTPUT_DIR = f"data/{LABEL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar MediaPipe y cámara
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print(f"Se capturarán {NUM_SAMPLES} muestras de la seña '{LABEL}' con duración de {DURATION} segundos cada una.")
print("Presioná ESPACIO para iniciar cada captura.")

for sample_num in range(1, NUM_SAMPLES + 1):
    input(f"\nMuestra {sample_num}/{NUM_SAMPLES} - Prepararse y presionar ENTER para comenzar...")
    frames = []
    start_time = time.time()

    while time.time() - start_time < DURATION:
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
            frames.append(keypoints)

        cv2.putText(frame, f"Grabando muestra {sample_num}/{NUM_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Captura de Seña con Movimiento", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Guardar muestra
    df = pd.DataFrame(frames)
    df.to_csv(os.path.join(OUTPUT_DIR, f"sample_{sample_num}.csv"), index=False)
    print(f"✅ Muestra {sample_num} guardada.")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Captura de datos para la seña '{LABEL}' completada.")

import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
from collections import deque, Counter

# ---------------------------- CONFIGURACIÓN ----------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "modelo.pkl")
BUFFER_SIZE = 15
STABLE_FRAMES = 4
CONF_THRESHOLD = 0.6

# ---------------------------- CARGA MODELO ----------------------------
if not os.path.exists(MODEL_PATH):
    print("No se encontró el modelo entrenado. Ejecuta el script de entrenamiento primero.")
    exit()

model = joblib.load(MODEL_PATH)
LABELS = model.classes_

# ---------------------------- MEDIAPIPE ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_processor = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---------------------------- FUNCIONES ----------------------------
def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords -= wrist
    size = np.linalg.norm(coords[9]-wrist)
    if size > 0:
        coords /= size
    return coords.flatten()

def get_closest_hand(landmarks_list):
    if not landmarks_list:
        return None
    min_dist, best = float('inf'), None
    for hand in landmarks_list:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        z_mean = np.mean([lm.z for lm in hand.landmark])
        dist = ((np.mean(xs)-0.5)*2 + (np.mean(ys)-0.5)*2)*0.5 + abs(z_mean)
        if dist < min_dist:
            min_dist, best = dist, hand
    return best

# ---------------------------- GUI ----------------------------
class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor de Lengua de Señas")
        self.root.state("zoomed")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.text_accum = ""
        self.register_enabled = True
        self.last_pred, self.stable_count = None, 0

        # Widgets
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.pred_var = StringVar()
        self.text_var = StringVar()
        tk.Label(root, textvariable=self.pred_var, font=("Arial", 20)).pack(pady=5)
        tk.Label(root, textvariable=self.text_var, font=("Arial", 28)).pack(pady=10)

        # Botones
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Pausar Registro", width=20, command=self.toggle_accum).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Borrar Registro", width=20, command=self.clear_accum).grid(row=0, column=1, padx=5)

        self.update_frame()

    def toggle_accum(self):
        self.register_enabled = not self.register_enabled

    def clear_accum(self):
        self.text_accum = ""
        self.text_var.set("")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_processor.process(rgb_frame)

        pred_label, prob = "Sin detección", 0.0
        if results.multi_hand_landmarks:
            hand_landmarks = get_closest_hand(results.multi_hand_landmarks)
            if hand_landmarks:
                features = normalize_landmarks(hand_landmarks)
                probs = model.predict_proba(features.reshape(1, -1))[0]
                pred_label = LABELS[np.argmax(probs)]
                self.buffer.append(pred_label)

                most_common, count = Counter(self.buffer).most_common(1)[0]
                if count / len(self.buffer) >= 0.5:
                    if pred_label == self.last_pred:
                        self.stable_count += 1
                    else:
                        self.stable_count, self.last_pred = 1, pred_label
                    if self.stable_count >= STABLE_FRAMES and self.register_enabled:
                        if not self.text_accum or self.text_accum[-1] != pred_label:
                            self.text_accum += pred_label

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        self.pred_var.set(f"Seña: {pred_label} ({prob:.2f})")
        self.text_var.set(self.text_accum)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

# ---------------------------- MAIN ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignApp(root)
    root.mainloop()
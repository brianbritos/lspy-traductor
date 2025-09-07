#!/usr/bin/env python3
"""
traductor.py - Predicción en vivo del modelo de señas
Uso:
    python traductor.py
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
from collections import deque, Counter

# ------------------- CONFIGURACIÓN -------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "modelo.pkl")

CONF_THRESHOLD = 0.8       # Umbral de confianza
BUFFER_SIZE = 5            # Suavizado de predicciones

# ------------------- CARGA DEL MODELO -------------------
if not os.path.exists(MODEL_PATH):
    print("❌ No se encontró el modelo entrenado. Ejecuta primero 'entrenador.py' para generarlo.")
    exit()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit()

LABELS = model.classes_

# ------------------- MEDIAPIPE -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ------------------- FUNCIONES -------------------
def extract_features(landmarks):
    """Convierte landmarks en vector plano"""
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords).flatten()

def predict_sign(features, buffer):
    """Predice seña con suavizado por buffer"""
    if features is None:
        return "Sin detección", 0.0

    features = features.reshape(1, -1)
    probs = model.predict_proba(features)[0]
    max_prob = np.max(probs)
    pred_label = LABELS[np.argmax(probs)]

    if max_prob >= CONF_THRESHOLD:
        buffer.append(pred_label)
        final_label = Counter(buffer).most_common(1)[0][0]
        return final_label, max_prob
    return "Sin detección", max_prob

# ------------------- GUI -------------------
class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor de Lengua de Señas")

        # Frame de video
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Label de predicción
        self.pred_var = StringVar()
        self.pred_label = tk.Label(root, textvariable=self.pred_var, font=("Arial", 20))
        self.pred_label.pack()

        # Texto acumulado
        self.text_var = StringVar()
        self.text_label = tk.Label(root, textvariable=self.text_var, font=("Arial", 24))
        self.text_label.pack()

        # Botones
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=10)

        self.toggle_button = tk.Button(self.buttons_frame, text="⏸️ Pausar registro", command=self.toggle_accum)
        self.toggle_button.grid(row=0, column=0, padx=5)

        self.clear_button = tk.Button(self.buttons_frame, text="🗑️ Borrar registro", command=self.clear_accum)
        self.clear_button.grid(row=0, column=1, padx=5)

        # Variables internas
        self.text_accum = ""
        self.register_enabled = True
        self.cap = cv2.VideoCapture(0)
        self.buffer = deque(maxlen=BUFFER_SIZE)

        self.update_frame()

    def toggle_accum(self):
        """Activa o desactiva el registro de letras"""
        self.register_enabled = not self.register_enabled
        if self.register_enabled:
            self.toggle_button.config(text="⏸️ Pausar registro")
        else:
            self.toggle_button.config(text="▶️ Reanudar registro")

    def clear_accum(self):
        """Limpia el texto acumulado"""
        self.text_accum = ""
        self.text_var.set("")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar mano
        results = hands.process(rgb_frame)
        pred_label, prob = "Sin detección", 0.0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_features(hand_landmarks)
                pred_label, prob = predict_sign(features, self.buffer)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar resultados
        if pred_label != "Sin detección":
            self.pred_var.set(f"Seña: {pred_label} ({prob:.2f})")
            if self.register_enabled:  # Solo acumula si está activado
                if len(self.text_accum) == 0 or self.text_accum[-1] != pred_label:
                    self.text_accum += pred_label
        else:
            self.pred_var.set("Sin detección")

        self.text_var.set(self.text_accum)

        # Convertir frame para Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignApp(root)
    root.mainloop()

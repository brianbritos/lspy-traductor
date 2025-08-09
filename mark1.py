import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

# ------------------- CARGA DE MODELO -------------------
# Se asume que ya entrenaste un RandomForest y lo guardaste como 'model.pkl'
try:
    model = joblib.load("model.pkl")
except:
    print("No se encontró el modelo entrenado. Entrena y guarda como model.pkl")
    exit()

# Lista de etiquetas (debe coincidir con el entrenamiento)
LABELS = model.classes_

# Umbral de confianza
CONF_THRESHOLD = 0.8

# ------------------- MEDIAPIPE -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ------------------- FUNCIONES -------------------
def extract_features(landmarks):
    """Extrae los landmarks en un vector 1D normalizado"""
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords).flatten()

def predict_sign(features):
    """Predice la seña y devuelve (etiqueta, confianza) o (None, 0) si es bajo el umbral"""
    if features is None:
        return None, 0.0
    features = features.reshape(1, -1)
    probs = model.predict_proba(features)[0]
    max_prob = np.max(probs)
    pred_label = LABELS[np.argmax(probs)]
    if max_prob >= CONF_THRESHOLD:
        return pred_label, max_prob
    return None, max_prob

# ------------------- GUI -------------------
class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lector de Lengua de Señas")
        
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.pred_var = StringVar()
        self.pred_label = tk.Label(root, textvariable=self.pred_var, font=("Arial", 20))
        self.pred_label.pack()

        self.text_var = StringVar()
        self.text_label = tk.Label(root, textvariable=self.text_var, font=("Arial", 24))
        self.text_label.pack()

        self.text_accum = ""  # Texto construido
        self.cap = cv2.VideoCapture(0)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con mediapipe
        results = hands.process(rgb_frame)
        pred_label, prob = None, 0.0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_features(hand_landmarks)
                pred_label, prob = predict_sign(features)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Mostrar resultados
        if pred_label is not None:
            self.pred_var.set(f"Seña: {pred_label} ({prob:.2f})")
            if len(self.text_accum) == 0 or (len(self.text_accum) > 0 and self.text_accum[-1] != pred_label):
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

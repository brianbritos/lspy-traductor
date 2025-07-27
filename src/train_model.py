# === train_model.py (corregido: columnas unificadas para evitar duplicados) ===
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
all_data = []
EXPECTED_FEATURES = 63
COLUMN_NAMES = [f'kp_{i}' for i in range(EXPECTED_FEATURES)] + ['label']

# Archivos .csv estáticos
for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    if file.endswith(".csv") and os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            if 'label' in df.columns and df.shape[1] == EXPECTED_FEATURES + 1:
                df.columns = COLUMN_NAMES  # Unificar nombres
                all_data.append(df)
            else:
                print(f"⚠️ Archivo ignorado (columnas incorrectas o sin 'label'): {file}")
        except Exception as e:
            print(f"⚠️ Error leyendo {file}: {e}")

# Carpetas con señas dinámicas
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file))
                    if df.shape[1] == EXPECTED_FEATURES:
                        descriptor = df.mean().tolist()
                        descriptor.append(folder)
                        df_sample = pd.DataFrame([descriptor], columns=COLUMN_NAMES)
                        all_data.append(df_sample)
                    else:
                        print(f"⚠️ Ignorado {file} en {folder}: columnas = {df.shape[1]}")
                except Exception as e:
                    print(f"⚠️ Error procesando {file} en {folder}: {e}")

if not all_data:
    print("❌ No se encontraron datos válidos para entrenar.")
    exit()

df = pd.concat(all_data, ignore_index=True)

# Validación final de columnas
if df.shape[1] != EXPECTED_FEATURES + 1:
    raise ValueError(f"❌ El dataset combinado tiene {df.shape[1]} columnas. Se esperaban {EXPECTED_FEATURES + 1} (63 + 'label').")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
print("✅ Modelo guardado como model.pkl (63 features verificados y consistentes)")

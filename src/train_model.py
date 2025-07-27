#train_model.py
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Leer todos los archivos CSV
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
all_data = []

for file in os.listdir(data_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_path, file))
        all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

# Separar X e y
X = df.drop('label', axis=1)
y = df['label']

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(model, "model.pkl")
print("✅ Modelo guardado como model.pkl")
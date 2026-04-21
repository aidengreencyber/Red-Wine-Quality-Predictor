import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. LOAD DATA
# -----------------------------
file_path = "C:\\Users\\agree\\Personal\\Personal\\AI-Code\\Red-Wine-Quality\\winequality-red.csv"
df = pd.read_csv(file_path, sep=";")

# -----------------------------
# 2. CREATE TARGET
# 1 = good quality (quality >= 6)
# 0 = not good quality (quality < 6)
# -----------------------------
df["good_quality"] = np.where(df["quality"] >= 6, 1, 0)

# -----------------------------
# 3. FEATURES AND LABEL
# -----------------------------
X = df.drop(columns=["quality", "good_quality"])
y = df["good_quality"]

# Save feature order for later use
feature_names = list(X.columns)

# -----------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 7. EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 8. SAVE MODEL FILES
# -----------------------------
os.makedirs("saved_models", exist_ok=True)

joblib.dump(model, "saved_models/model.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(feature_names, "saved_models/feature_names.pkl")

print("\nSaved model, scaler, and feature names to /saved_models")
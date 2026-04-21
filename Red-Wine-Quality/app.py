from pathlib import Path
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine measurements below and click **Predict Quality**.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model_path = MODEL_DIR / "model.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"
features_path = MODEL_DIR / "feature_names.pkl"

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

if not scaler_path.exists():
    st.error(f"Scaler file not found: {scaler_path}")
    st.stop()

if not features_path.exists():
    st.error(f"Feature names file not found: {features_path}")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

# -----------------------------
# INPUT FORM
# -----------------------------
with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
    citric_acid = st.number_input("Citric Acid", value=0.00)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    ph = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

    submitted = st.form_submit_button("Predict Quality")

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:
    input_values = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]])

    input_scaled = scaler.transform(input_values)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("Predicted Quality: GOOD")
    else:
        st.error("Predicted Quality: NOT GOOD")

    st.write(f"Confidence that it is good quality: **{probability:.2%}**")

# -----------------------------
# OPTIONAL: SHOW FEATURE ORDER
# -----------------------------
with st.expander("See feature order used by the model"):
    st.write(feature_names)

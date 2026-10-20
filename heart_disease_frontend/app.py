import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "KNN_hear (1).pkl")
    scaler = joblib.load(BASE_DIR / "scaler (1).pkl")
    columns = joblib.load(BASE_DIR / "columns (1).pkl")
    return model, scaler, list(columns)

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter the health measurements below to get a model-based prediction.")
st.warning("This tool is for educational purposes only. It does not diagnose heart disease or replace medical advice.")

try:
    model, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error("Could not load the model files. Keep app.py and all three .pkl files in the same folder.")
    st.caption(f"Technical detail: {e}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Personal and health details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], format_func=lambda x: {
            "ATA": "ATA — Atypical Angina", "NAP": "NAP — Non-Anginal Pain",
            "TA": "TA — Typical Angina", "ASY": "ASY — Asymptomatic"}[x])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=800, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=250, value=150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"], format_func=lambda x: "No" if x == "N" else "Yes")
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary", use_container_width=True)

if submitted:
    # Build exactly the one-hot encoded feature columns used during model training.
    row = {
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": int(sex == "M"),
        "ChestPainType_ATA": int(chest_pain == "ATA"),
        "ChestPainType_NAP": int(chest_pain == "NAP"),
        "ChestPainType_TA": int(chest_pain == "TA"),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg == "ST"),
        "ExerciseAngina_Y": int(exercise_angina == "Y"),
        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up"),
    }
    # Age is intentionally not included because it is absent from the saved training columns.
    # Reorder columns to match columns.pkl exactly.
    input_df = pd.DataFrame([[row.get(col, 0) for col in feature_columns]], columns=feature_columns)

    try:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            classes = list(model.classes_)
            if 1 in classes:
                risk_score = float(probabilities[classes.index(1)]) * 100
                st.metric("Model-estimated positive class probability", f"{risk_score:.1f}%")
        st.divider()
        if prediction == 1:
            st.error("The model predicts a higher likelihood of heart disease (class 1).")
            st.write("Please discuss these results and your symptoms with a qualified healthcare professional.")
        else:
            st.success("The model predicts no heart disease detected (class 0).")
            st.write("A negative prediction does not rule out heart disease. Seek medical advice if you have symptoms or concerns.")
        with st.expander("See the encoded input sent to the model"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as e:
        st.error("Prediction failed. Check that the model and scaler were trained on the same features and in the same order.")
        st.caption(f"Technical detail: {e}")

st.caption("Model: K-Nearest Neighbors (KNN). Predictions depend on the training data and are not medical diagnoses.")

# ============================================================
# app.py
# Cloud-based Health Dashboard for Medical Professionals
# with Charts & Patient Record Logging
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ============================================================
# Page Configuration (MUST be first Streamlit command)
# ============================================================
st.set_page_config(page_title="Cloud Health Dashboard", layout="wide")

# ============================================================
# Load trained model and scaler
# ============================================================
MODEL_PATH = os.path.join("models", "heart_disease_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or Scaler file not found! Please place your .pkl files in /models folder.")
    st.stop()
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("✅ Model and Scaler loaded successfully!")

# ============================================================
# App Title
# ============================================================
st.title("☁️ Cloud-based Health Dashboard for Medical Professionals")
st.markdown("### AI-powered Risk Prediction for Busy Urban Hospitals")
st.markdown("---")

# ============================================================
# Sidebar: Patient Inputs
# ============================================================
st.sidebar.header("🧍‍♀️ Patient Information")

age = st.sidebar.slider("Age (years)", 18, 100, 45)
sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
cp = st.sidebar.selectbox("Chest Pain Type (cp)",
                          ("Typical Angina (1)", "Atypical Angina (2)", "Non-anginal (3)", "Asymptomatic (4)"))
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ("False", "True"))
restecg = st.sidebar.selectbox("Resting ECG Result", ("Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"))
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ("No", "Yes"))
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", ("Upsloping (1)", "Flat (2)", "Downsloping (3)"))
ca = st.sidebar.slider("Major Vessels Colored by Fluoroscopy", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia (thal)", ("Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"))

# ============================================================
# Convert Inputs to Numeric
# ============================================================
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "True" else 0
exang = 1 if exang == "Yes" else 0
cp = int(cp.split("(")[1][0])
restecg = int(restecg.split("(")[1][0])
slope = int(slope.split("(")[1][0])
thal = int(thal.split("(")[1][0])

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
input_data_scaled = scaler.transform(input_data)

input_df = pd.DataFrame(input_data, columns=[
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
])

# ============================================================
# Live Vitals Dashboard
# ============================================================
st.subheader("📊 Patient Vital Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🩸 Blood Pressure", value=f"{trestbps} mmHg")
    st.metric(label="🧬 Cholesterol", value=f"{chol} mg/dL")
with col2:
    st.metric(label="❤️ Max Heart Rate", value=f"{thalach} bpm")
    st.metric(label="📉 ST Depression", value=f"{oldpeak}")
with col3:
    st.metric(label="🧍 Age", value=age)
    st.metric(label="🧠 Chest Pain Type", value=cp)

st.markdown("---")

# ============================================================
# Predict Button
# ============================================================
if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_data_scaled)
    risk = "⚠️ High Risk of Heart Disease" if prediction[0] == 1 else "✅ Low Risk - Stable Condition"

    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error(risk)
    else:
        st.success(risk)

    # ------------------------------
    # Save patient data to CSV
    # ------------------------------
    csv_path = os.path.join("data", "patient_records.csv")
    os.makedirs("data", exist_ok=True)

    input_df["prediction"] = prediction[0]
    input_df["risk_level"] = "High" if prediction[0] == 1 else "Low"

    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        updated_data = pd.concat([existing_data, input_df], ignore_index=True)
    else:
        updated_data = input_df

    updated_data.to_csv(csv_path, index=False)
    st.success("🗂 Patient record saved successfully!")

    st.markdown("#### Patient Data Summary")
    st.dataframe(input_df)

# ============================================================
# Live Charts
# ============================================================
st.markdown("---")
st.subheader("📈 Visual Analytics")

csv_path = os.path.join("data", "patient_records.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Heart Rate vs Age**")
        fig, ax = plt.subplots()
        ax.scatter(df["age"], df["thalach"], c=df["prediction"], cmap="coolwarm", label="Risk")
        ax.set_xlabel("Age")
        ax.set_ylabel("Max Heart Rate")
        st.pyplot(fig)

    with col2:
        st.markdown("**Cholesterol vs Blood Pressure**")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["chol"], df["trestbps"], c=df["prediction"], cmap="viridis")
        ax2.set_xlabel("Cholesterol")
        ax2.set_ylabel("Resting BP")
        st.pyplot(fig2)

    st.markdown("#### Risk Level Distribution")
    st.bar_chart(df["risk_level"].value_counts())

else:
    st.info("No patient records yet. Predictions will appear here once available.")

st.markdown("---")
st.caption("AIML For Health Care| SAKEC | Cloud Health Dashboard Project")

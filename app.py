import streamlit as st
import numpy as np
import joblib

# Load Model
model  = joblib.load("model/logistic_regression.pkl")
scaler = joblib.load("model/scaler.pkl")

# Prediction Function
def predict(data):
    scaled      = scaler.transform([data])
    prediction  = model.predict(scaled)
    probability = model.predict_proba(scaled)
    return prediction[0], probability[0]

# Feature Engineering
def get_features(pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, dpf, age):
    bmi_cat = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    age_grp = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    glc_lvl = 0 if glucose < 100 else 1 if glucose < 126 else 2
    return [pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, bmi_cat, age_grp, glc_lvl]

# UI
st.set_page_config(page_title="Diabetes Classifier", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Classifier")
st.markdown("ادخل بيانات المريض وهنقولك هل عنده خطر إصابة بالسكري أم لأ")
st.divider()

# Input Fields
col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.slider("Pregnancies",    0,   17,  3)
    glucose        = st.slider("Glucose",        0,  200, 120)
    blood_pressure = st.slider("Blood Pressure", 0,  122,  70)
    skin_thickness = st.slider("Skin Thickness", 0,  100,  20)
    insulin        = st.slider("Insulin",        0,  850,  80)

with col2:
    bmi = st.slider("BMI",              0.0, 67.0, 25.0)
    dpf = st.slider("Diabetes Pedigree", 0.0, 2.5,  0.5)
    age = st.slider("Age",               21,   81,  30)

st.divider()

# Predict Button
if st.button("🔍 Predict", use_container_width=True):
    features   = get_features(pregnancies, glucose, blood_pressure,
                               skin_thickness, insulin, bmi, dpf, age)
    pred, prob = predict(features)

    st.divider()

    if pred == 1:
        st.error("### 🩺 Diabetes Risk Detected")
    else:
        st.success("### ✅ No Diabetes Risk")

    col1, col2 = st.columns(2)
    col1.metric("No Diabetes", f"{prob[0]:.2%}")
    col2.metric("Diabetes",    f"{prob[1]:.2%}")

    st.progress(float(prob[1]))
    st.caption("كلما زاد البار كلما زاد خطر الإصابة بالسكري")
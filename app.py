import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 Diabetes Prediction App")

st.write(
    "Enter the patient information below to estimate "
    "the predicted probability of the diabetes outcome."
)

st.divider()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=32.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=33,
        step=1
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Apply the same preprocessing used during training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability of Outcome = 1
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"Positive predicted outcome\n\n"
            f"Estimated probability: {probability:.2%}"
        )

    else:

        st.success(
            f"Negative predicted outcome\n\n"
            f"Estimated probability: {probability:.2%}"
        )

    st.info(
        "This application is for educational and demonstration "
        "purposes only and is not a medical diagnosis."
    )
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")

# Input layout
col1, col2 = st.columns(2)

with col1:
    passenger_id = st.number_input(
        "Passenger ID", min_value=1, max_value=10000, value=1, step=1
    )
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 0, 80, 30)

with col2:
    fare = st.number_input(
        "Fare", min_value=0.0, max_value=500.0, value=50.0
    )
    sex = st.selectbox("Sex", ["Male", "Female"])

# Encode sex
sex_encoded = 1 if sex == "Male" else 0

# Prediction
if st.button("Predict Survival"):
    # ✅ MUST match training feature names & order
    input_df = pd.DataFrame(
        [[passenger_id, pclass, age, fare, sex_encoded]],
        columns=["PassengerId", "Pclass", "Age", "Fare", "Sex_encoded"]
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader("Prediction Results")

    if prediction == 1:
        st.success(
            f"✅ Passenger {passenger_id} survived "
            f"(Probability: {probability:.2%})"
        )
    else:
        st.error(
            f"❌ Passenger {passenger_id} did NOT survive "
            f"(Probability: {probability:.2%})"
        )

    st.write("---")
    st.write("**Input Details:**")
    st.write(f"- Passenger ID: {passenger_id}")
    st.write(f"- Class: {pclass}")
    st.write(f"- Age: {age}")
    st.write(f"- Fare: ${fare:.2f}")
    st.write(f"- Sex: {sex}")

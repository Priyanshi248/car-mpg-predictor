import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("car_mpg_model.pkl")
encoder = joblib.load("origin_encoder.pkl")

# Title
st.title("🚗 Car MPG Predictor")

st.write("Enter car details to predict mileage")

# Input fields
cylinders = st.number_input("Cylinders", 1, 12, 4)
displacement = st.number_input("Displacement", 0.0, 500.0, 150.0)
horsepower = st.number_input("Horsepower", 0.0, 300.0, 100.0)
weight = st.number_input("Weight", 0.0, 5000.0, 2500.0)
acceleration = st.number_input("Acceleration", 0.0, 30.0, 15.0)
model_year = st.number_input("Model Year", 0, 100, 70)

origin = st.selectbox("Origin", encoder.classes_)

# Button
if st.button("Predict MPG"):
    origin_val = encoder.transform([origin])[0]

    input_data = pd.DataFrame([[
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        origin_val
    ]], columns=[
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    ])

    result = model.predict(input_data)[0]

    st.success(f"Predicted MPG: {result:.2f}")
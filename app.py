import streamlit as st
from joblib import load
import pandas as pd

model = load('car_price_model.joblib')

# ---------------- Streamlit UI ---------------- #

st.title("Used Car Price Prediction")
st.write("Enter the features to predict the estimated car price")

# User Inputs
make_year = st.number_input("Make Year", min_value=1980, max_value=2025, value=2015)
mileage_kmpl = st.number_input("Mileage (km/l)", min_value=0.0, value=15.0)
engine_cc = st.number_input("Engine Size (cc)", min_value=500, value=1500)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
owner_count = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1)
brand = st.selectbox("Brand", ["Chevrolet", "Honda", "BMW", "Hyundai", "Nissan", "Toyota", "Ford", "Kia", "Other"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
color = st.selectbox("Color", ["White", "Black", "Blue", "Red", "Grey", "Silver", "Other"])
service_history = st.selectbox("Service History", ["Full", "None", "Partial"])
accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=20, value=0)
insurance_valid = st.selectbox("Insurance Valid", ["Yes", "No"])

# Predict button
if st.button("Predict Price"):
    # Prepare input as a DataFrame
    input_data = pd.DataFrame([{
        'make_year': make_year,
        'mileage_kmpl': mileage_kmpl,
        'engine_cc': engine_cc,
        'fuel_type': fuel_type,
        'owner_count': owner_count,
        'brand': brand,
        'transmission': transmission,
        'color': color,
        'service_history': service_history,
        'accidents_reported': accidents_reported,
        'insurance_valid': insurance_valid
    }])

    # Make prediction
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ${predicted_price:,.2f}")

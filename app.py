import streamlit as st
import pickle
import numpy as np

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please run model.py first to train and save the model.")
    st.stop()

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

# Input fields in two columns for better layout
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income (in $10,000s)", 0.0, 15.0, 3.0, 0.1)
    HouseAge = st.slider("House Age", 1, 50, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0, 0.1)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)

with col2:
    Population = st.slider("Population", 100.0, 10000.0, 1000.0, 50.0)
    AveOccup = st.slider("Average Occupancy", 1.0, 5.0, 3.0, 0.1)
    Latitude = st.slider("Latitude", 32.0, 42.0, 36.0, 0.1)
    Longitude = st.slider("Longitude", -124.0, -114.0, -120.0, 0.1)

# Prediction
if st.button("Predict Price"):
    try:
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, 
                              Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted House Price: ${prediction * 100000:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
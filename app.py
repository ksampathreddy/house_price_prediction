import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

# Input fields
MedInc = st.slider("Median Income (in $10,000s)", 0.0, 15.0, 3.0)
HouseAge = st.slider("House Age", 1, 50, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 100.0, 10000.0, 1000.0)
AveOccup = st.slider("Average Occupancy", 1.0, 5.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted House Price: ${prediction * 100000:,.2f}")

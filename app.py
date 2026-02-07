import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("lr.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 Boston House Price Prediction App")

st.write("Enter all 13 feature values below:")

# --- Input fields for all 13 features ---
crim = st.number_input("CRIM: Crime rate per capita", value=0.0)
zn = st.number_input("ZN: Residential land zoned", value=0.0)
indus = st.number_input("INDUS: Non-retail business acres", value=0.0)
chas = st.number_input("CHAS: Charles River dummy (0 or 1)", value=0.0)
nox = st.number_input("NOX: Nitric oxide concentration", value=0.0)
rm = st.number_input("RM: Average number of rooms", value=0.0)
age = st.number_input("AGE: Owner-occupied units built prior to 1940", value=0.0)
dis = st.number_input("DIS: Distance to employment centres", value=0.0)
rad = st.number_input("RAD: Accessibility to highways", value=0.0)
tax = st.number_input("TAX: Property tax rate", value=0.0)
ptratio = st.number_input("PTRATIO: Pupil-teacher ratio", value=0.0)
b = st.number_input("B: Proportion of Black population", value=0.0)
lstat = st.number_input("LSTAT: % lower status population", value=0.0)

# --- Prediction button ---
if st.button("Predict House Price"):

    # Combine all inputs into array (13 features)
    input_data = np.array([[crim, zn, indus, chas, nox,
                            rm, age, dis, rad, tax,
                            ptratio, b, lstat]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"🏡 Predicted House Price: {prediction[0]:.2f}")
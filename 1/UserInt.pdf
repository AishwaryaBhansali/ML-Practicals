# placement_ui.py
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Placement Predictor - 2024-25 Batch")

# Input fields
tenth = st.slider("10th %", 0, 100, 75)
twelfth = st.slider("12th %", 0, 100, 75)
fe = st.slider("FE %", 0, 100, 70)
se = st.slider("SE %", 0, 100, 72)
te = st.slider("TE %", 0, 100, 74)
certs = st.slider("Certifications Completed", 0, 10, 2)
projects = st.slider("Projects Completed", 0, 10, 2)
internships = st.slider("Internships Done", 0, 5, 1)

# Predict button
if st.button("Predict Placement"):
    input_data = np.array([[tenth, twelfth, fe, se, te, certs, projects, internships]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("✅ Likely to be Placed!")
    else:
        st.error("❌ Not likely to be Placed.")

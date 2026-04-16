import streamlit as st
import pickle
import numpy as np

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="AI Purchase Predictor", page_icon="💰", layout="centered")

# -----------------------
# Load model + scaler
# -----------------------
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------
# Title
# -----------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💰 AI Purchase Prediction App</h1>", unsafe_allow_html=True)

st.write("### Predict whether a user will purchase a product based on profile data")

st.markdown("---")

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("User Input Panel")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
salary = st.sidebar.number_input("Estimated Salary", 15000, 150000, 50000)

# -----------------------
# Encoding
# -----------------------
gender_val = 0 if gender == "Male" else 1

# -----------------------
# Prediction
# -----------------------
if st.button("🚀 Predict Now"):

    input_data = np.array([[gender_val, age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    # -----------------------
    # Output
    # -----------------------
    st.markdown("---")

    if prediction[0] == 1:
        st.success("🎯 User WILL PURCHASE the product!")
    else:
        st.error("❌ User will NOT purchase the product.")

    st.markdown("---")

    st.info("Model: KNN | Dataset: Social Media Ads | Accuracy ~91%")

# -----------------------
# Footer
# -----------------------
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Purchase Predictor",
    page_icon="💰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .sub {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>💰 AI Purchase Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Predict if a user will buy based on profile data</div>", unsafe_allow_html=True)

st.write("")

# ---------------- INPUT BOXES ----------------
st.markdown("### 🧾 Enter User Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)

with col2:
    salary = st.number_input("Estimated Salary", 15000, 150000, 50000)

# ---------------- ENCODE ----------------
gender_val = 0 if gender == "Male" else 1

# ---------------- PREDICT ----------------
st.write("")

if st.button("🚀 Predict Now"):

    input_data = np.array([[gender_val, age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.markdown("---")

    # ---------------- OUTPUT ----------------
    if prediction[0] == 1:
        st.success("🎯 YES! User will PURCHASE the product")
        st.balloons()
    else:
        st.error("❌ NO! User will NOT purchase")

    st.markdown("---")

    st.info("📊 Model: KNN | Accuracy ~91% | Dataset: Social Ads")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit | AI ML Project
    </p>
    """,
    unsafe_allow_html=True
)
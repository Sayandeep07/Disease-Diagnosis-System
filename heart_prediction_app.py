import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("heart.csv")

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Streamlit UI
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓")
st.title("💓 Heart Disease Prediction App")
st.markdown("Enter the following details to check for heart disease risk:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=50)

sex = st.radio("Sex", ["Female (0)", "Male (1)"])
sex = 0 if "Female" in sex else 1

cp = st.selectbox("Chest Pain Type (cp)", [
    "0: Typical angina", 
    "1: Atypical angina", 
    "2: Non-anginal pain", 
    "3: Asymptomatic"
])
cp = int(cp[0])  # Extract number

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No (0)", "Yes (1)"])
fbs = 1 if "Yes" in fbs else 0

restecg = st.selectbox("Resting ECG Results (restecg)", [
    "0: Normal", 
    "1: ST-T wave abnormality", 
    "2: Left ventricular hypertrophy"
])
restecg = int(restecg[0])

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=210, value=150)

exang = st.radio("Exercise Induced Angina?", ["No (0)", "Yes (1)"])
exang = 1 if "Yes" in exang else 0

oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia (thal)", [
    "0: Unknown", 
    "1: Normal", 
    "2: Fixed defect", 
    "3: Reversible defect"
])
thal = int(thal[0])  # Extract number

# Predict button
if st.button("🔍 Predict"):
    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease.")

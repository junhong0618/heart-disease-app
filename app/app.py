import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load models & scaler
models = {
    "Random Forest": joblib.load("app/random_forest.pkl"),
    "Gradient Boosting": joblib.load("app/gradient_boosting.pkl"),
    "Logistic Regression": joblib.load("app/logistic_regression.pkl"),
    "Support Vector Machine": joblib.load("app/support_vector_machine.pkl"),
    "K-Nearest Neighbors": joblib.load("app/k-nearest_neighbors.pkl")
}
scaler = joblib.load("app/scaler.pkl")

# Title
st.title("💓 Heart Disease Prediction Dashboard")

# Sidebar - User Input
st.sidebar.header("User Input Features")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 20, 100, 50)
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4])
    smoker = st.sidebar.selectbox("Current Smoker", ["Yes", "No"])
    cigs_per_day = st.sidebar.slider("Cigarettes per Day", 0, 50, 10)
    bp_meds = st.sidebar.selectbox("On BP Medication", ["Yes", "No"])
    prevalent_stroke = st.sidebar.selectbox("History of Stroke", ["Yes", "No"])
    prevalent_hyp = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
    diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"])
    tot_chol = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
    sys_bp = st.sidebar.slider("Systolic BP", 90, 200, 120)
    dia_bp = st.sidebar.slider("Diastolic BP", 60, 130, 80)
    bmi = st.sidebar.slider("BMI", 15, 50, 25)
    heart_rate = st.sidebar.slider("Heart Rate", 40, 150, 70)
    glucose = st.sidebar.slider("Glucose Level", 50, 300, 100)

    # Convert categorical inputs
    gender = 1 if gender == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    bp_meds = 1 if bp_meds == "Yes" else 0
    prevalent_stroke = 1 if prevalent_stroke == "Yes" else 0
    prevalent_hyp = 1 if prevalent_hyp == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0

    # Create DataFrame
    input_data = pd.DataFrame(
        [[gender, age, education, smoker, cigs_per_day, bp_meds, prevalent_stroke, prevalent_hyp, diabetes,
          tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose]],
        columns=["Gender", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke",
                 "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    )

    # Scale numerical values
    numerical_features = ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    return input_data

# Get user input
input_df = user_input()

# Prediction Button
if st.sidebar.button("Predict Heart Disease Risk"):
    st.subheader("🔍 Prediction Results from Multiple Models")
    for model_name, model in models.items():
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] * 100 if hasattr(model, "predict_proba") else "N/A"
        
        if prediction[0] == 1:
            st.error(f"⚠️ {model_name}: High Risk ({probability}%)")
        else:
            st.success(f"✅ {model_name}: Low Risk ({probability}%)")

st.write("💡 Regular exercise and a healthy diet help reduce heart disease risk!")

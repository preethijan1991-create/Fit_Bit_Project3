import pandas as pd
import numpy as np
import pickle
import streamlit as st

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("C:/Users/Shiva S R/OneDrive/Desktop/Fitbit_dataset - Fitbit_dataset.csv (1).csv")

# ---------------------------
# Outlier Capping
# ---------------------------
num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = np.clip(df[col], lower, upper)

# ---------------------------
# Encoding
# ---------------------------
encoder = OrdinalEncoder()

df[['Gender','Workout_Type']] = encoder.fit_transform(
    df[['Gender','Workout_Type']]
)

# ---------------------------
# Features & Target
# ---------------------------
X = df.drop("Calories_Burned (kcal)", axis=1)
y = df["Calories_Burned (kcal)"]

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------
# Train Model
# ---------------------------
model = LinearRegression()

model.fit(X_scaled, y)

# ---------------------------
# Save Files
# ---------------------------
pickle.dump(model,
            open("calorie_model.pkl","wb"))

pickle.dump(scaler,
            open("scaler.pkl","wb"))

pickle.dump(encoder,
            open("encoder.pkl","wb"))

print("Model Saved Successfully")




# Load assets
model = pickle.load(open("calorie_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))

st.set_page_config(
    page_title="Calories Burn Predictor",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Calories Burn Prediction")
st.write("Enter your workout details")

# -----------------------
# User Inputs
# -----------------------

age = st.number_input("Age", 10, 100, 25)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

weight = st.number_input(
    "Weight (kg)",
    value=70.0
)

height = st.number_input(
    "Height (m)",
    value=1.75
)

max_bpm = st.number_input("Max BPM", value=180)

avg_bpm = st.number_input("Avg BPM", value=120)

resting_bpm = st.number_input("Resting BPM", value=70)

session_duration = st.number_input(
    "Session Duration (hours)",
    value=1.0
)

fat_percentage = st.number_input(
    "Fat Percentage",
    value=20.0
)

water_intake = st.number_input(
    "Water Intake (liters)",
    value=2.0
)

workout_frequency = st.number_input(
    "Workout Frequency (days/week)",
    value=4
)

experience_level = st.number_input(
    "Experience Level",
    value=2
)

bmi = st.number_input(
    "BMI",
    value=22.5
)

base_met = st.number_input(
    "Base MET",
    value=3.0
)

hr_intensity = st.number_input(
    "HR Intensity",
    value=0.7
)

effective_met = st.number_input(
    "Effective MET",
    value=5.0
)
workout_type = st.selectbox(
    "Workout Type",
    ["Cardio", "HIIT", "Strength", "Yoga"]
)

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Calories Burned"):

    input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Weight (kg)': [weight],
    'Height (m)': [height],
    'Max_BPM': [max_bpm],
    'Avg_BPM': [avg_bpm],
    'Resting_BPM': [resting_bpm],
    'Session_Duration (hours)': [session_duration],
    'Workout_Type': [workout_type],
    'Fat_Percentage': [fat_percentage],
    'Water_Intake (liters)': [water_intake],
    'Workout_Frequency (days/week)': [workout_frequency],
    'Experience_Level': [experience_level],
    'BMI': [bmi],
    'Base_MET': [base_met],
    'HR_Intensity': [hr_intensity],
    'Effective_MET': [effective_met]
})

    input_df[['Gender', 'Workout_Type']] = encoder.transform(
        input_df[['Gender', 'Workout_Type']]
    )

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)

    st.success(
        f"🔥 Estimated Calories Burned: {prediction[0]:.2f} kcal"
    )
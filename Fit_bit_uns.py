import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "C:/Users/Shiva S R/OneDrive/Desktop/Fitbit_dataset - Fitbit_dataset.csv (1).csv"
)

# ==================================================
# PREPROCESSING
# ==================================================

df_cluster = df.drop("Workout_Type", axis=1)

le = LabelEncoder()

df_cluster["Gender"] = le.fit_transform(
    df_cluster["Gender"]
)

X = df_cluster

# ==================================================
# SCALING
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# PCA
# ==================================================

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

# ==================================================
# FIND BEST K
# ==================================================

scores = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    scores.append(score)

best_k = np.argmax(scores) + 2

# ==================================================
# TRAIN FINAL KMEANS
# ==================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)

df["Cluster"] = clusters

# ==================================================
# STREAMLIT UI
# ==================================================

st.set_page_config(
    page_title="Workout Pattern Clustering",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ Workout Pattern Clustering")

st.write(
    "Identify hidden workout behavior patterns using PCA and KMeans."
)

# ==================================================
# SHOW MODEL PERFORMANCE
# ==================================================

final_score = silhouette_score(
    X_pca,
    clusters
)

st.metric(
    "Silhouette Score",
    round(final_score, 3)
)

# ==================================================
# USER INPUTS
# ==================================================

st.subheader("Enter Workout Details")

age = st.number_input(
    "Age",
    10,
    100,
    25
)

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

max_bpm = st.number_input(
    "Max BPM",
    value=180
)

avg_bpm = st.number_input(
    "Average BPM",
    value=120
)

resting_bpm = st.number_input(
    "Resting BPM",
    value=70
)

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
    min_value=1,
    max_value=5,
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

calories = st.number_input(
    "Calories Burned (kcal)",
    value=350
)

# ==================================================
# CLUSTER PREDICTION
# ==================================================

if st.button("Find My Cluster"):

    gender_encoded = le.transform(
        [gender]
    )[0]

    input_df = pd.DataFrame({

        "Age":[age],

        "Gender":[gender_encoded],

        "Weight (kg)":[weight],

        "Height (m)":[height],

        "Max_BPM":[max_bpm],

        "Avg_BPM":[avg_bpm],

        "Resting_BPM":[resting_bpm],

        "Session_Duration (hours)":[session_duration],

        "Fat_Percentage":[fat_percentage],

        "Water_Intake (liters)":[water_intake],

        "Workout_Frequency (days/week)":[workout_frequency],

        "Experience_Level":[experience_level],

        "BMI":[bmi],

        "Base_MET":[base_met],

        "HR_Intensity":[hr_intensity],

        "Effective_MET":[effective_met],

        "Calories_Burned (kcal)":[calories]

    })

    scaled_data = scaler.transform(
        input_df
    )

    pca_data = pca.transform(
        scaled_data
    )

    cluster = kmeans.predict(
        pca_data
    )[0]

    st.success(
        f"You belong to Cluster {cluster}"
    )

    # ==========================================
    # INTERPRET CLUSTER
    # ==========================================

    if cluster == 0:

        st.info(
            "🔵 Low Intensity Workout Pattern"
        )

    elif cluster == 1:

        st.info(
            "🟢 Moderate Intensity Workout Pattern"
        )

    else:

        st.info(
            "🔴 High Intensity Workout Pattern"
        )

# ==================================================
# CLUSTER SUMMARY
# ==================================================

st.subheader("Cluster Analysis")

cluster_centroids = df.groupby(
    "Cluster"
)[[
    "Avg_BPM",
    "Max_BPM",
    "HR_Intensity",
    "Effective_MET",
    "Calories_Burned (kcal)",
    "Session_Duration (hours)"
]].mean().round(2)

st.dataframe(cluster_centroids)

# ==================================================
# EXPERIENCE LEVEL ANALYSIS
# ==================================================

st.subheader(
    "Experience Level vs Cluster"
)

experience_table = pd.crosstab(
    df["Cluster"],
    df["Experience_Level"]
)

st.dataframe(
    experience_table
)

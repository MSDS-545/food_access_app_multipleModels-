import os
import streamlit as st
import requests
import pandas as pd
import altair as alt

# Get API URL from env var or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Food Access Model Prediction")
st.write("Explore data, select model, and submit input features to predict LILA (Low Income & Low Access)")

# 1. Load data for EDA
@st.cache_data
def load_data():
    df = pd.read_csv("FoodAccessResearchAtlas.csv", low_memory=False)
    return df
df = load_data()

# 2. EDA visualisations
feature_cols = ['HUNVFlag', 'PovertyRate', 'LA1and10']
eda_df = df[feature_cols].copy()

st.subheader("Exploratory Data Analysis (EDA)")

for col in feature_cols:
    st.write(f"### {col}")
    hist = alt.Chart(eda_df).mark_bar().encode(
        alt.X(f"{col}:Q", bin=True),
        alt.Y('count()')
    ).properties(width=600, height=200)
    st.altair_chart(hist)

st.write("### Pairplot (simplified)")
scatter = alt.Chart(eda_df).mark_circle(opacity=0.3).encode(
    x=f"{feature_cols[0]}:Q",
    y=f"{feature_cols[1]}:Q",
    color=f"{feature_cols[2]}:Q"
).properties(width=600, height=400)
st.altair_chart(scatter)

# 3. Model selection & input
model_choice = st.selectbox(
    "Choose prediction model",
    ("logistic", "rf", "xgb"),
    key="model_choice_select"
)

st.subheader("Input Features for Prediction")
hunv_flag    = st.number_input("HUNVFlag (households with no vehicle)", min_value=0.0, value=0.0, key="hunv_flag_input")
poverty_rate = st.number_input("PovertyRate (% below poverty level)", min_value=0.0, max_value=100.0, value=25.0, key="poverty_rate_input")
la1and10     = st.number_input("LA1and10 (low‐access flag: 1 mile urban / 10 miles rural)", min_value=0.0, value=0.0, key="la1and10_input")

if st.button("Predict", key="predict_button"):
    endpoint = "/predict/simple"
    url = f"{API_URL}{endpoint}"
    payload = {
        "HUNVFlag":   hunv_flag,
        "PovertyRate": poverty_rate,
        "LA1and10":    la1and10,
        "model_type":  model_choice
    }

    st.write("### Debug Info")
    st.write("Request URL:", url)
    st.write("Payload:", payload)

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        st.success(
            f"Model: {result['model_used']}\n"
            f"Predicted Class: {result['predicted_class']} (1 = LILA)\n"
            f"Probability: {result['probability']:.2f}"
        )
    except Exception as err:
        st.error(f"Error: {err}")

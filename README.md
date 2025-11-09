Food Access Prediction App – Tutorial

A full-stack machine learning web application to predict Low-Income & Low-Access (LILA) census tracts using the Food Access Research Atlas from the USDA Economic Research Service (ERS).
Built with:

Backend: FastAPI (model-serving)

Frontend: Streamlit (UI + EDA)

Models supported: Logistic Regression, Random Forest, XGBoost

Deployment: Locally and via Docker & Docker Compose

Table of Contents

Prerequisites

Notebook: Data Preparation & Modeling

Backend Service (FastAPI)

Frontend UI (Streamlit)

Docker & Docker-Compose Setup

Running & Testing Locally / With Docker

Key Concepts & Syntax Explained

Troubleshooting & Common Issues

Next Steps & Extensions

1. Prerequisites {#prerequisites}

Before you begin, ensure you have:

Python 3.10+ installed

(Optional) A virtual environment to isolate dependencies

Docker & Docker Compose installed (if you prefer containerised deployment)

The dataset from the Food Access Research Atlas downloaded (CSV or XLSX)

2. Notebook: Data Preparation & Modeling {#notebook-data-preparation-modelling}

In this notebook you will:

Load the dataset

Select three features (HUNVFlag, PovertyRate, LA1and10)

Preprocess data (imputation, scaling)

Train three models (Logistic Regression, Random Forest, XGBoost)

Evaluate models and save pipelines & trained models

Code snippet: Logistic Regression
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
```
# 1. Load your data
df = pd.read_csv("FoodAccessResearchAtlas.csv", low_memory=False)
target_col = "LILATracts_1And10"
feature_cols = ['HUNVFlag', 'PovertyRate', 'LA1and10']
df = df[[target_col] + feature_cols].dropna().copy()
X_raw = df[feature_cols]
y = df[target_col].astype(int)

# 2. Impute & Scale
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X_raw)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4. Train Logistic Regression model
logreg = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
logreg.fit(X_train, y_train)

# 5. Evaluate
y_pred  = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

print("Accuracy:",     accuracy_score(y_test, y_pred))
print("Precision:",    precision_score(y_test, y_pred))
print("Recall:",       recall_score(y_test, y_pred))
print("F1-score:",     f1_score(y_test, y_pred))
print("ROC AUC:",      roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save pipelines & model
joblib.dump(imputer, "imputer_logistic.pkl")
joblib.dump(scaler, "scaler_logistic.pkl")
joblib.dump(logreg, "model_logistic.pkl")
print("Saved logistic regression model and preprocessing files.")

Code snippet: Random Forest
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Load and prepare
df = pd.read_csv("FoodAccessResearchAtlas.csv", low_memory=False)
target_col  = "LILATracts_1And10"
feature_cols = ['HUNVFlag', 'PovertyRate', 'LA1and10']
df = df[[target_col] + feature_cols].dropna().copy()
X_raw = df[feature_cols]
y     = df[target_col].astype(int)

# Preprocess
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X_raw)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("Accuracy:",     accuracy_score(y_test, y_pred))
print("Precision:",    precision_score(y_test, y_pred))
print("Recall:",       recall_score(y_test, y_pred))
print("F1-score:",     f1_score(y_test, y_pred))
print("ROC AUC:",      roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(imputer, "imputer_rf.pkl")
joblib.dump(scaler, "scaler_rf.pkl")
joblib.dump(rf, "model_rf.pkl")
print("Saved Random Forest model and preprocessing files.")

Code snippet: XGBoost
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Load & prepare
df = pd.read_csv("FoodAccessResearchAtlas.csv", low_memory=False)
target_col  = "LILATracts_1And10"
feature_cols = ['HUNVFlag', 'PovertyRate', 'LA1and10']
df = df[[target_col] + feature_cols].dropna().copy()
X_raw = df[feature_cols]
y     = df[target_col].astype(int)

# Preprocess
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X_raw)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
xgb_clf.fit(X_train, y_train)

# Evaluate
y_pred  = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("Accuracy:",     accuracy_score(y_test, y_pred))
print("Precision:",    precision_score(y_test, y_pred))
print("Recall:",       recall_score(y_test, y_pred))
print("F1-score:",     f1_score(y_test, y_pred))
print("ROC AUC:",      roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(imputer, "imputer_xgb.pkl")
joblib.dump(scaler, "scaler_xgb.pkl")
joblib.dump(xgb_clf, "model_xgb.pkl")
print("Saved XGBoost model and preprocessing files.")

3. Backend Service (FastAPI) {#backend-service-fastapi}

This is the code in backend/app.py. It loads the saved models and offers a prediction endpoint.

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Food Access Prediction API")

# Load pipelines & models
models = {
    "logistic": {
        "imputer": joblib.load("models/imputer_logistic.pkl"),
        "scaler":  joblib.load("models/scaler_logistic.pkl"),
        "model":   joblib.load("models/model_logistic.pkl")
    },
    "rf": {
        "imputer": joblib.load("models/imputer_rf.pkl"),
        "scaler":  joblib.load("models/scaler_rf.pkl"),
        "model":   joblib.load("models/model_rf.pkl")
    },
    "xgb": {
        "imputer": joblib.load("models/imputer_xgb.pkl"),
        "scaler":  joblib.load("models/scaler_xgb.pkl"),
        "model":   joblib.load("models/model_xgb.pkl")
    },
}

class Features3(BaseModel):
    HUNVFlag:    float
    PovertyRate: float
    LA1and10:    float

class PredictionResponse(BaseModel):
    predicted_class: int
    probability:     float
    model_used:      str

@app.post("/predict/simple", response_model=PredictionResponse)
def predict_simple(
    data: Features3,
    model_type: str = Query("logistic", enum=["logistic","rf","xgb"])
):
    if model_type not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
    pipeline = models[model_type]
    try:
        X_new = np.array([[data.HUNVFlag, data.PovertyRate, data.LA1and10]])
        X_imp = pipeline["imputer"].transform(X_new)
        X_scaled = pipeline["scaler"].transform(X_imp)
        prob = pipeline["model"].predict_proba(X_scaled)[0][1]
        pred = int(pipeline["model"].predict(X_scaled)[0])
        return PredictionResponse(predicted_class=pred, probability=prob, model_used=model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```
Explanation:

FastAPI() creates the application instance.

The models dictionary maps each model type (“logistic”, “rf”, “xgb”) to its preprocessing pipeline and trained model.

Features3 defines the schema of input features.

PredictionResponse defines the output schema including probability and which model was used.

The /predict/simple endpoint expects a POST request with JSON matching Features3 and a query parameter model_type.

Internally: build X_new → apply imputer & scaler → predict probability & class → return the result.

4. Frontend UI (Streamlit) {#frontend-ui-streamlit}

This is the code in frontend/app.py. It provides both EDA visualisations and a prediction interface.
```
import os
import streamlit as st
import requests
import pandas as pd
import altair as alt

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Food Access Model Prediction")
st.write("Explore data, select model & submit inputs to predict LILA (Low Income & Low Access)")

@st.cache_data
def load_data():
    return pd.read_csv("FoodAccessResearchAtlas.csv", low_memory=False)
df = load_data()

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

st.write("### Pair-plot (simplified)")
scatter = alt.Chart(eda_df).mark_circle(opacity=0.3).encode(
    x=f"{feature_cols[0]}:Q",
    y=f"{feature_cols[1]}:Q",
    color=f"{feature_cols[2]}:Q"
).properties(width=600, height=400)
st.altair_chart(scatter)

model_choice = st.selectbox(
    "Choose prediction model",
    ("logistic", "rf", "xgb"),
    key="model_choice_select"
)

st.subheader("Input Features for Prediction")
hunv_flag    = st.number_input("HUNVFlag (households with no vehicle)", min_value=0.0, value=0.0, key="hunv_flag_input")
poverty_rate = st.number_input("PovertyRate (% below poverty level)", min_value=0.0, max_value=100.0, value=25.0, key="poverty_rate_input")
la1and10     = st.number_input("LA1and10 (low-access flag: 1 mile urban / 10 miles rural)", min_value=0.0, value=0.0, key="la1and10_input")

if st.button("Predict", key="predict_button"):
    endpoint = "/predict/simple"
    url      = f"{API_URL}{endpoint}"
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
        resp   = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        st.success(
            f"Model: {result['model_used']}\n"
            f"Predicted Class: {result['predicted_class']} (1 = LILA)\n"
            f"Probability: {result['probability']:.2f}"
        )
    except Exception as err:
        st.error(f"Error: {err}")

```
Explanation:

Imports: os, streamlit, requests, pandas, altair.

API_URL is set via environment variable or defaults to http://localhost:8000.

load_data() reads the dataset once, cached for performance.

EDA section shows histograms + a scatter plot for the selected features.

Model selection dropdown: user picks "logistic", "rf", or "xgb".

Input widgets for each feature.

On clicking Predict, the payload is constructed and sent to the backend. Debug info shows URL & payload for troubleshooting.

5. Docker & Docker-Compose Setup {#docker-docker-compose-setup}

These files allow you to containerise the backend & frontend and run them together.

Backend Dockerfile
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

Frontend Dockerfile
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
    environment:
      - LOG_LEVEL=info

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - API_URL=http://backend:8000


Explanation:

FROM python:3.10-slim: lightweight Python base image.

WORKDIR /app: sets working directory inside container.

COPY commands: bring code and model files into the container.

RUN pip install: install dependencies.

EXPOSE: tells Docker which port the container listens on.

CMD: default command when container starts.

In Compose: services: defines two containers — backend & frontend.

ports: maps host ports to container ports (e.g., "8000:8000").

depends_on: ensures frontend starts after backend.

environment: sets environment variables inside containers (e.g., API_URL=http://backend:8000).

6. Running & Testing Locally / With Docker {#running-testing}
Locally (without Docker)
# Run backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# In new terminal: run frontend
cd ../frontend
pip install -r requirements.txt
streamlit run app.py


Backend docs: http://localhost:8000/docs

Frontend UI: http://localhost:8501

With Docker Compose
docker compose up --build


After build, visit: http://localhost:8501

The UI will call the backend at http://backend:8000 via Docker’s internal network.

Test the flow (curl examples)
# Logistic Regression
curl -X POST "http://localhost:8000/predict/simple?model_type=logistic" \
     -H "Content-Type: application/json" \
     -d '{"HUNVFlag": 0.0, "PovertyRate": 25.0, "LA1and10": 1.0}'

# Random Forest
curl -X POST "http://localhost:8000/predict/simple?model_type=rf" \
     -H "Content-Type: application/json" \
     -d '{"HUNVFlag": 0.0, "PovertyRate": 25.0, "LA1and10": 1.0}'

# XGBoost
curl -X POST "http://localhost:8000/predict/simple?model_type=xgb" \
     -H "Content-Type: application/json" \
     -d '{"HUNVFlag": 0.0, "PovertyRate": 25.0, "LA1and10": 1.0}'

Running Locally (without Docker)

Set up your backend
Open a terminal and navigate to the backend/ directory:

cd backend


Create and activate a virtual environment (recommended):

python3 -m venv .venv  
source .venv/bin/activate        # On Windows: .venv\Scripts\activate


Install dependencies from requirements.txt:

pip install -r requirements.txt


Run your FastAPI backend with Uvicorn:

uvicorn app:app --reload --host 0.0.0.0 --port 8000


This starts the server locally — you’ll see something like:

INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


In your browser you can open the interactive API docs at:
http://localhost:8000/docs

Set up your frontend
Open another terminal and navigate to the frontend/ directory:

cd ../frontend


Create/activate a virtual environment (optional but recommended):

python3 -m venv .venv  
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


After running, open your browser to:
http://localhost:8501

There you’ll see the UI with EDA visualisations and the prediction form.

Test the flow

In the UI, select one of the models (logistic / rf / xgb), input values for the features (HUNVFlag, PovertyRate, LA1and10), and click Predict.

The UI will send a request to http://localhost:8000/predict/simple with the selected model type and your input features.

If everything is wired correctly, you should get a JSON response (via UI) showing the model used, predicted class and probability.

You can also test via curl from a terminal:

curl -X POST "http://localhost:8000/predict/simple?model_type=logistic" \
     -H "Content-Type: application/json" \
     -d '{"HUNVFlag":0.0,"PovertyRate":25.0,"LA1and10":1.0}'


Debug if needed

If the UI shows unexpected results (e.g., class = 0 or probability ≈ 0), check the debug info printed in the UI (payload & URL).

Confirm the backend terminal logs what Features3 receives and which model was used.

Ensure your frontend’s API_URL is set to http://localhost:8000 (not http://backend:8000, which is for Docker).

Ensure the backend is listening on 0.0.0.0 so it’s reachable externally from the host.

7. Key Concepts & Syntax Explained {#key-concepts}

Pipeline: sequence of transformations (imputer → scaler) then model. Ensures that inference uses identical preprocessing as training.

FastAPI route decorator: @app.post("/predict/simple") defines a POST endpoint at that path.

Pydantic BaseModel: used for validating and parsing request payloads (e.g., Features3).

Streamlit widgets: st.selectbox, st.number_input, st.button provide interactive UI controls.

Docker basics:

Dockerfile – defines how to build the image.

docker-compose.yml – manages multi-container setups (services, networks, volumes).

ports: "8000:8000" – maps host port to container port.

volumes: – shared folders between host and container (e.g., for model files).

8. Troubleshooting & Common Issues {#troubleshooting}
Issue	Cause	Fix
“Pipeline expects XXX features, got 3”	Model was trained on more features than sent	Use correct number of features or retrain with fewer features
UI result always class = 0 & probability ≈ 0	Payload keys mismatch or wrong endpoint used	Debug payload & URL in UI, verify backend logs
Frontend container cannot resolve backend host	Docker network mis-configured or using localhost	Ensure API_URL=http://backend:8000 for containerised frontend
xgboost==1.8.5 not found	Version not available on PyPI for your platform	Use a valid version like xgboost==3.1.1
Backend unreachable from frontend in Docker	Ports or host binding mis-set	Backend must listen on 0.0.0.0, exposed port correct
9. Next Steps & Extensions {#next-steps}

Add more features from the dataset (vehicle access counts, distance to supermarkets, demographic metrics)

Create a /predict/full endpoint for a full-feature model and update UI accordingly

Deploy to cloud services (AWS ECS/Fargate, Azure Web App, GCP Cloud Run)

Add authentication, logging, monitoring (e.g., with Prometheus, Grafana)

Expand EDA: correlation matrices, feature importance, SHAP values

Build automated model-retraining pipelines (CI/CD)
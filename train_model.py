import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# -------------------
# LOAD DATA
# -------------------

df = pd.read_csv(
    "C:\ds and AI\ALL DATASETS\semiconductor_wafer_defect_dataset.csv"
)

# -------------------
# ENCODING
# -------------------

le = LabelEncoder()

df["process_step_encoded"] = le.fit_transform(
    df["process_step"]
)

# -------------------
# FEATURES
# -------------------

features = [
    "temperature_c",
    "pressure_torr",
    "gas_flow_sccm",
    "etch_rate_nm_min",
    "voltage_v",
    "current_ma",
    "process_step_encoded"
]

X = df[features]

# -------------------
# SCALING
# -------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------
# ISOLATION FOREST
# -------------------

iso = IsolationForest(
    n_estimators=500,
    contamination=0.002,
    random_state=42
)

iso.fit(X_scaled)

# -------------------
# ONE CLASS SVM
# -------------------

svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.002
)

svm.fit(X_scaled)

# -------------------
# SAVE
# -------------------

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    iso,
    "models/isolation_forest.pkl"
)

joblib.dump(
    svm,
    "models/oneclass_svm.pkl"
)

print("Models Saved Successfully")
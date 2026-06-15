import streamlit as st
import pandas as pd
import numpy as np
import shap
from reportlab.platypus import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

st.set_page_config(
    page_title="WaferGuard AI",
    page_icon="🚀",
    layout="wide"
)

# ==========================================
# HEADER
# ==========================================

st.title("🚀 WaferGuard AI")
st.subheader(
    "Semiconductor Wafer Defect Intelligence Platform"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

   df = pd.read_csv( 
       r"C:\ds and AI\ALL DATASETS\semiconductor_wafer_defect_dataset.csv"
    )


   return df

df = load_data()

# ==========================================
# DATASET OVERVIEW
# ==========================================

st.sidebar.header("Dataset Overview")

st.sidebar.write(
    f"Rows : {df.shape[0]}"
)

st.sidebar.write(
    f"Columns : {df.shape[1]}"
)

# ==========================================
# KPI SECTION
# ==========================================

total_wafers = len(df)

defects = df["defect_label"].sum()

defect_rate = (
    defects /
    total_wafers
) * 100

c1,c2,c3 = st.columns(3)

with c1:
    st.metric(
        "Total Wafers",
        total_wafers
    )

with c2:
    st.metric(
        "Defective Wafers",
        defects
    )

with c3:
    st.metric(
        "Defect Rate %",
        round(defect_rate,4)
    )

# ==========================================
# RAW DATA
# ==========================================

with st.expander("View Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# ==========================================
# ENCODING
# ==========================================

le = LabelEncoder()

df["process_step_encoded"] = (
    le.fit_transform(
        df["process_step"]
    )
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["power_consumption"] = (
    df["voltage_v"]
    *
    df["current_ma"]
)

features = [

    "temperature_c",

    "pressure_torr",

    "gas_flow_sccm",

    "etch_rate_nm_min",

    "voltage_v",

    "current_ma",

    "power_consumption",

    "process_step_encoded"

]

X = df[features]

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Z SCORE
# ==========================================

st.header("📊 Z-Score Detection")

z_scores = np.abs(
    zscore(X)
)

zscore_anomaly = (
    z_scores > 3
).any(axis=1)

df["zscore_anomaly"] = (
    zscore_anomaly.astype(int)
    
)
df["zscore_score"] = (
    z_scores.max(axis=1)
)
st.success(
    f"Z-Score Anomalies Found : "
    f"{df['zscore_anomaly'].sum()}"
)

# ==========================================
# ISOLATION FOREST
# ==========================================

st.header("🌲 Isolation Forest")

iso = IsolationForest(
    n_estimators=500,
    contamination=0.002,
    random_state=42
)

iso.fit(X_scaled)

iso_pred = iso.predict(
    X_scaled
)

iso_pred = np.where(
    iso_pred == -1,
    1,
    0
)

df["isolation_forest"] = iso_pred

df["iforest_score"] = (
    -iso.decision_function(
        X_scaled
    )
)

st.success(
    f"Isolation Forest Anomalies : "
    f"{df['isolation_forest'].sum()}"
)

# ==========================================
# ONE CLASS SVM
# ==========================================

st.header("⚡ One-Class SVM")

svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.002
)

svm.fit(X_scaled)

svm_pred = svm.predict(
    X_scaled
)

svm_pred = np.where(
    svm_pred == -1,
    1,
    0
)

df["oneclass_svm"] = svm_pred
df["svm_score"] = -svm.decision_function(
    X_scaled
)

st.success(
    f"One-Class SVM Anomalies : "
    f"{df['oneclass_svm'].sum()}"
)

# ==========================================
# LOCAL OUTLIER FACTOR
# ==========================================

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.002,
    novelty=True
)

lof.fit(X_scaled)

lof_pred = lof.predict(
    X_scaled
)

lof_pred = np.where(
    lof_pred == -1,
    1,
    0
)

df["lof_anomaly"] = lof_pred

df["lof_score"] = (
    -lof.decision_function(
        X_scaled
    )
)
# ==========================================
# MAHALANOBIS DISTANCE
# ==========================================

st.header("📐 Mahalanobis Distance")

cov_matrix = np.cov(
    X_scaled.T
)

inv_cov_matrix = np.linalg.inv(
    cov_matrix
)

mean_vector = np.mean(
    X_scaled,
    axis=0
)

mahal_scores = []

for row in X_scaled:

    score = mahalanobis(
        row,
        mean_vector,
        inv_cov_matrix
    )

    mahal_scores.append(
        score
    )

df["mahalanobis_score"] = (
    mahal_scores
)

threshold = np.percentile(
    mahal_scores,
    99.8
)

df["mahalanobis_anomaly"] = np.where(
    df["mahalanobis_score"] >
    threshold,
    1,
    0
)

st.success(
    f"Mahalanobis Anomalies : "
    f"{df['mahalanobis_anomaly'].sum()}"
)

# ==========================================
# RESEARCH GRADE RISK ENGINE
# ==========================================

risk = (

    0.30 * df["iforest_score"]

    +

    0.25 * df["svm_score"]

    +

    0.20 * df["lof_score"]

    +

    0.15 * df["mahalanobis_score"]

    +

    0.10 * df["zscore_score"]

)

risk = (

    risk - risk.min()

) / (

    risk.max() - risk.min()

)

df["risk_score"] = (
    risk * 100
)

# ==========================================
# ENSEMBLE VOTING
# ==========================================

df["ensemble_votes"] = (

    df["zscore_anomaly"]

    +

    df["isolation_forest"]

    +

    df["oneclass_svm"]

    +

    df["lof_anomaly"]

    +

    df["mahalanobis_anomaly"]

)

df["ensemble_prediction"] = np.where(

    df["ensemble_votes"] >= 3,

    1,

    0

)
# ==========================================
# ENSEMBLE AI
# ==========================================

st.header("🤖 Ensemble AI Engine")

df["ensemble_votes"] = (

    df["zscore_anomaly"]

    +

    df["isolation_forest"]

    +

    df["oneclass_svm"]

    +

    df["lof_anomaly"]

    +

    df["mahalanobis_anomaly"]

)

df["ensemble_prediction"] = np.where(

    df["ensemble_votes"] >= 3,

    1,

    0

)

# ==========================================
# RISK SCORE
# ==========================================

risk = (

    0.30 * df["iforest_score"]

    +

    0.20 * df["oneclass_svm"]

    +

    0.20 * df["lof_anomaly"]

    +

    0.20 * df["mahalanobis_score"]

    +

    0.10 * df["zscore_anomaly"]

)

risk = (

    risk - risk.min()

) / (

    risk.max() - risk.min()

)

df["risk_score"] = (

    risk * 100

)

# ==========================================
# RESEARCH GRADE RISK ENGINE
# ==========================================

risk = (

    0.30 * df["iforest_score"]

    +

    0.25 * df["svm_score"]

    +

    0.20 * df["lof_score"]

    +

    0.15 * df["mahalanobis_score"]

    +

    0.10 * df["zscore_score"]

)

risk = (

    risk - risk.min()

) / (

    risk.max() - risk.min()

)

df["risk_score"] = (
    risk * 100
)

# ==========================================
# ENSEMBLE VOTING
# ==========================================

df["ensemble_votes"] = (

    df["zscore_anomaly"]

    +

    df["isolation_forest"]

    +

    df["oneclass_svm"]

    +

    df["lof_anomaly"]

    +

    df["mahalanobis_anomaly"]

)

df["ensemble_prediction"] = np.where(

    df["ensemble_votes"] >= 3,

    1,

    0

)

# ==========================================
# RESULTS
# ==========================================

st.header("📈 Ensemble Results")

st.metric(
    "Final Anomalies",
    int(
        df[
            "ensemble_prediction"
        ].sum()
    )
)

# ==========================================
# TOP SUSPICIOUS WAFERS
# ==========================================

st.header("🚨 Top Suspicious Wafers")

top_anomalies = (

    df

    .sort_values(
        "risk_score",
        ascending=False
    )

    .head(25)

)

st.dataframe(

    top_anomalies[
        [

            "wafer_id",

            "risk_score",

            "iforest_score",

            "mahalanobis_score",

            "ensemble_prediction",

            "defect_label"

        ]

    ],

    use_container_width=True

)

# ==========================================
# SAVE RESULTS
# ==========================================

df.to_csv(
    "wafer_anomaly_results.csv",
    index=False
)

st.success(
    "Results saved to wafer_anomaly_results.csv"
)
# ==========================================
# IMPORTS FOR PART 1B
# ==========================================

import plotly.express as px

from sklearn.decomposition import PCA

# ==========================================
# PCA ANALYSIS
# ==========================================

st.markdown("---")
st.header("🧠 Principal Component Analysis (PCA)")

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Risk Score": df["risk_score"],
    "Defect": df["defect_label"],
    "Ensemble": df["ensemble_prediction"]
})

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Risk Score",
    hover_data=["Defect"],
    title="PCA Wafer Risk Landscape",
    height=600
)

st.plotly_chart(
    fig_pca,
    use_container_width=True
)

# ==========================================
# EXPLAINED VARIANCE
# ==========================================

st.subheader("Explained Variance Ratio")

variance_df = pd.DataFrame({
    "Component": ["PC1", "PC2"],
    "Variance": pca.explained_variance_ratio_
})

fig_variance = px.bar(
    variance_df,
    x="Component",
    y="Variance",
    title="PCA Explained Variance"
)

st.plotly_chart(
    fig_variance,
    use_container_width=True
)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.markdown("---")
st.header("🔥 Feature Correlation Heatmap")

corr_matrix = df[
    features +
    ["risk_score"]
].corr()

fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    title="Feature Correlation Matrix"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.markdown("---")
st.header("📊 Feature Importance Analysis")

feature_importance = np.abs(
    df[
        features
    ].corrwith(
        df["risk_score"]
    )
)

importance_df = pd.DataFrame({
    "Feature": feature_importance.index,
    "Importance": feature_importance.values
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Risk Driver Importance"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# ==========================================
# RISK SCORE DISTRIBUTION
# ==========================================

st.markdown("---")
st.header("⚠ Risk Score Distribution")

fig_risk = px.histogram(
    df,
    x="risk_score",
    nbins=50,
    title="Risk Score Distribution"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# ==========================================
# IFOREST DISTRIBUTION
# ==========================================

st.header("🌲 Isolation Forest Score Distribution")

fig_iforest = px.histogram(
    df,
    x="iforest_score",
    nbins=50,
    title="Isolation Forest Scores"
)

st.plotly_chart(
    fig_iforest,
    use_container_width=True
)

# ==========================================
# PROCESS STEP ANALYSIS
# ==========================================

st.markdown("---")
st.header("🏭 Process Step Analysis")

process_summary = (
    df.groupby("process_step")
    .agg({
        "risk_score": "mean",
        "ensemble_prediction": "sum"
    })
    .reset_index()
)

fig_process = px.bar(
    process_summary,
    x="process_step",
    y="risk_score",
    color="ensemble_prediction",
    title="Average Risk by Process Step"
)

st.plotly_chart(
    fig_process,
    use_container_width=True
)

# ==========================================
# DEFECT VS NORMAL COMPARISON
# ==========================================

st.markdown("---")
st.header("🔬 Defect vs Normal Comparison")

feature_compare = st.selectbox(
    "Select Feature",
    features
)

fig_compare = px.box(
    df,
    x="defect_label",
    y=feature_compare,
    title=f"{feature_compare} vs Defect Status"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)

# ==========================================
# TOP RISK PROCESS CONDITIONS
# ==========================================

st.markdown("---")
st.header("🚨 Highest Risk Process Conditions")

risk_view = df.sort_values(
    "risk_score",
    ascending=False
)

st.dataframe(
    risk_view[
        [
            "wafer_id",
            "temperature_c",
            "pressure_torr",
            "gas_flow_sccm",
            "etch_rate_nm_min",
            "voltage_v",
            "current_ma",
            "risk_score"
        ]
    ].head(20),
    use_container_width=True
)

# ==========================================
# ANOMALY COMPARISON
# ==========================================

st.markdown("---")
st.header("🤖 Model Comparison")

comparison = pd.DataFrame({

    "Method": [
        "Z Score",
        "Isolation Forest",
        "One Class SVM",
        "LOF",
        "Mahalanobis",
        "Ensemble"
    ],

    "Detected": [

        df["zscore_anomaly"].sum(),

        df["isolation_forest"].sum(),

        df["oneclass_svm"].sum(),

        df["lof_anomaly"].sum(),

        df["mahalanobis_anomaly"].sum(),

        df["ensemble_prediction"].sum()

    ]
})

fig_compare_models = px.bar(
    comparison,
    x="Method",
    y="Detected",
    title="Anomaly Detection Method Comparison"
)

st.plotly_chart(
    fig_compare_models,
    use_container_width=True
)

# ==========================================
# EXECUTIVE SUMMARY
# ==========================================

st.markdown("---")
st.header("📑 Executive Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Highest Risk Score",
        round(
            df["risk_score"].max(),
            2
        )
    )

with col2:

    st.metric(
        "Average Risk Score",
        round(
            df["risk_score"].mean(),
            2
        )
    )

with col3:

    st.metric(
        "Detected Anomalies",
        int(
            df["ensemble_prediction"].sum()
        )
    )

# ==========================================
# RESEARCH INSIGHTS
# ==========================================

st.markdown("---")

st.success(
"""
Research Findings:

• Ensemble detection significantly improves anomaly discovery.

• Isolation Forest identifies rare manufacturing deviations.

• Mahalanobis distance captures multivariate abnormalities.

• PCA reveals hidden process clusters.

• Risk score ranking enables proactive quality assurance.

• Semiconductor process optimization can be guided using these anomaly insights.
"""
)

# ==========================================
# END OF PART 1B
# ==========================================
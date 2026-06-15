# 🚀 WaferGuard AI

## Research-Grade Semiconductor Wafer Anomaly Detection Platform

### Overview

WaferGuard AI is an advanced anomaly detection and manufacturing intelligence platform designed for semiconductor wafer quality monitoring. The project combines multiple unsupervised machine learning algorithms and statistical techniques to identify abnormal process behavior, rank manufacturing risks, and support proactive defect prevention.

This system simulates a real-world semiconductor manufacturing analytics workflow used in smart factories and Industry 4.0 environments.

---

## Business Problem

Semiconductor fabrication involves highly sensitive manufacturing processes where minor process deviations can result in defective wafers, yield loss, and significant financial impact.

Traditional threshold-based monitoring often fails to capture complex multivariate anomalies.

WaferGuard AI addresses this challenge by leveraging ensemble anomaly detection techniques to identify hidden process abnormalities before defects occur.

---

## Dataset Information

### Features

| Feature          | Description             |
| ---------------- | ----------------------- |
| wafer_id         | Unique wafer identifier |
| temperature_c    | Process temperature     |
| pressure_torr    | Chamber pressure        |
| gas_flow_sccm    | Gas flow rate           |
| etch_rate_nm_min | Etch rate               |
| voltage_v        | Applied voltage         |
| current_ma       | Current consumption     |
| process_step     | Manufacturing stage     |
| defect_label     | Actual defect indicator |

### Dataset Size

* Records: 5,000
* Features: 9
* Defective Wafers: 7
* Defect Rate: 0.14%

---

## Machine Learning Techniques

### Statistical Methods

* Z-Score Analysis
* Mahalanobis Distance

### Unsupervised Machine Learning

* Isolation Forest
* One-Class SVM
* Local Outlier Factor (LOF)

### Ensemble Intelligence Engine

Multiple anomaly detection models are combined through a weighted risk scoring framework to improve robustness and reliability.

---

## Risk Scoring Framework

Risk Score is calculated using a weighted ensemble approach:

Risk =

30% Isolation Forest

25% One-Class SVM

20% LOF

15% Mahalanobis Distance

10% Z-Score

The final score is normalized to generate a risk index ranging from 0–100.

---

## Analytics Dashboard Features

### Manufacturing Intelligence

* Wafer Risk Ranking
* Defect Monitoring
* Process Step Analysis
* Outlier Investigation

### Visual Analytics

* PCA Visualization
* Correlation Heatmap
* Risk Distribution Analysis
* Feature Importance Dashboard

### Executive Reporting

* Top Suspicious Wafers
* Risk Score Dashboard
* Anomaly Detection Summary
* Manufacturing Insights

---

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* SciPy

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Application Framework

* Streamlit

---

## Installation

```bash
git clone <repository-url>

cd WaferGuard-AI

pip install -r requirements.txt

streamlit run app.py
```

---

## Project Structure

```text
WaferGuard-AI/

├── app.py
├── requirements.txt
├── semiconductor_wafer_defect_dataset.csv
├── wafer_anomaly_results.csv
├── models/
├── assets/
└── README.md
```

---

## Key Outcomes

* Built a research-grade anomaly detection framework
* Developed an ensemble manufacturing risk engine
* Identified process abnormalities using multiple detection methods
* Created interactive semiconductor quality intelligence dashboards
* Demonstrated practical Industry 4.0 analytics capabilities

---

## Future Enhancements

* SHAP Explainability
* Deep Autoencoder Anomaly Detection
* Real-Time Manufacturing Monitoring
* API Deployment
* MLOps Pipeline Integration
* Predictive Yield Optimization

---

## Author
m viswa teja reddy


Designed and developed as an advanced AI, Machine Learning, and Manufacturing Analytics portfolio project demonstrating practical applications of unsupervised learning in semiconductor quality assurance and industrial intelligence systems.

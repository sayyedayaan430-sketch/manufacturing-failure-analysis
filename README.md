🏭 Manufacturing Failure Analysis

A data-driven project to analyze, visualize, and predict equipment failures in manufacturing using Machine Learning. 


📌 Table of Contents
Overview
Features
Project Structure
Tech Stack
Dataset
Getting Started
How to Run
How It Works
Visualizations
Model Performance
Contributing
License
🔍 Overview
Manufacturing equipment failures cause costly downtime and production losses. This project analyzes historical machine data to:
Identify patterns that lead to failures
Visualize failure trends across machines and time
Predict future failures before they happen using ML
Example:
Input  : Temperature=320°C, Vibration=0.85, Pressure=142 PSI, Runtime=4200hrs
Output : ⚠️ HIGH RISK OF FAILURE (87% confidence)
✨ Features
✅ Full data cleaning & preprocessing pipeline
✅ Exploratory Data Analysis (EDA) with rich visualizations
✅ Failure frequency analysis by machine, type, and time
✅ ML model to predict failure probability
✅ Feature importance chart — see what causes failures most
✅ Professional charts saved as PNG files
✅ Modular, well-commented codebase

🛠️ Tech Stack
Category
Tools
Language
Python 3.9+
Data Analysis
Pandas, NumPy
Visualization
Matplotlib, Seaborn
ML Model
scikit-learn (Random Forest)
Web App
Streamlit
Serialization
Pickle

📁 Dataset
Recommended free datasets:
Dataset
Source
Description
AI4I 2020 Predictive Maintenance
UCI / Kaggle
Machine sensor data with failure labels
Microsoft Azure Predictive Maintenance
Kaggle
Telemetry, errors, failures data
SECOM Manufacturing
UCI ML Repository
Semiconductor manufacturing signals

▶️ How to Run
Step 1 — Train the model (only once)
python src/train.py
Step 2 — Launch the web app
streamlit run app.py
Step 3 — Enter machine sensor values and click "Predict Failure" 🎉
💡 Or predict directly from terminal:
python src/predict.py --temp 320 --vibration 0.85 --pressure 142

⚙️ How It Works
Raw Sensor Data (CSV)
       │
       ▼
┌─────────────────────────┐
│   Data Cleaning         │  → Handle missing values,
│   (preprocess.py)       │    remove outliers, encode categories
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   EDA & Visualization   │  → Failure trends, correlation heatmap,
│   (visualize.py)        │    feature distributions
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Feature Engineering   │  → Select important features,
│                         │    scale numerical values
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Random Forest Model   │  → Train classifier to predict
│   (train.py)            │    failure vs no failure
└─────────────────────────┘
       │
       ▼
  Failure Prediction + Confidence Score

📊 Visualizations
The project generates the following charts saved in outputs/charts/:
Chart
Description
failure_distribution.png
Pie/bar chart of failure vs normal
failure_by_type.png
Breakdown of failure types
correlation_heatmap.png
Feature correlation matrix
feature_importance.png
Top features causing failures
failure_over_time.png
Failure frequency trend over time

📈 Model Performance
Metric
Score
Accuracy
~96%
Precision
~91%
Recall
~88%
F1 Score
~89%
ROC-AUC
~97%
⚠️ Scores are approximate and depend on the dataset used.

Cloud-Native Predictive Maintenance Platform

Overview

A cloud-native machine learning platform that predicts industrial machine failures before they occur using sensor data. The system combines XGBoost-based failure prediction, anomaly detection, time-series analysis, AWS cloud services, and a real-time monitoring dashboard to help reduce unplanned downtime and maintenance costs.

Key Features

- Predict machine failure probability using machine-learning models
- Detect abnormal machine behavior using Isolation Forest
- Estimate Remaining Useful Life (RUL)
- Perform automated feature engineering on sensor data
- Compare multiple ML algorithms
- Provide explainable predictions using SHAP
- Deploy trained models as cloud-based APIs
- Monitor predictions and system performance
- Visualize machine health through an interactive dashboard

Machine Learning Models

Task| Model
Failure Classification| XGBoost
Baseline Classification| Random Forest
High-performance Classification| LightGBM
Anomaly Detection| Isolation Forest
RUL Prediction| XGBoost Regressor / LSTM
Explainability| SHAP

Dataset

The project can use publicly available predictive-maintenance datasets such as:

- AI4I 2020 Predictive Maintenance Dataset
- NASA C-MAPSS Dataset

Typical features include temperature, rotational speed, torque, tool wear, pressure, and machine operating conditions.

System Architecture

Machine / Dataset
       ↓
   AWS IoT Core
       ↓
    Amazon S3
       ↓
Data Processing
       ↓
Feature Engineering
       ↓
Amazon SageMaker
       ↓
ML Model Training
       ↓
SageMaker Model Endpoint
       ↓
     FastAPI
       ↓
Angular Dashboard
       ↓
   CloudWatch

Technology Stack

Machine Learning

- Python
- Scikit-learn
- XGBoost
- LightGBM
- TensorFlow/PyTorch
- SHAP

Backend

- FastAPI
- REST APIs

Frontend

- Angular

Database

- PostgreSQL

Cloud

- AWS IoT Core
- Amazon S3
- AWS Lambda
- Amazon SageMaker
- API Gateway
- Amazon EC2/ECS
- Amazon CloudWatch

DevOps

- Docker
- GitHub Actions

ML Workflow

1. Collect sensor data.
2. Clean and preprocess the dataset.
3. Perform feature engineering.
4. Split data into training and testing sets.
5. Train multiple ML models.
6. Tune hyperparameters using Optuna/GridSearchCV.
7. Evaluate models using Precision, Recall, F1-score and ROC-AUC.
8. Select and register the best-performing model.
9. Deploy the model using AWS SageMaker.
10. Serve predictions through FastAPI.
11. Display machine health and predictions on the Angular dashboard.
12. Monitor the deployed system and retrain the model when necessary.

Example Prediction

Machine ID: M-1024

Failure Probability: 87.4%
Health Status: HIGH RISK

Top Contributing Factors:
1. Tool Wear
2. Rotational Speed
3. Process Temperature
4. Torque

Recommended Action:
Schedule preventive maintenance.

Project Goals

- Reduce unexpected machine downtime
- Enable predictive rather than reactive maintenance
- Provide real-time machine health monitoring
- Build a scalable ML deployment pipeline
- Demonstrate production-oriented ML and cloud engineering

Future Scope

- Real-time streaming sensor data
- Edge-based ML inference
- Automated model retraining
- Model drift detection
- Digital twin integration
- Multi-machine fleet prediction
- Automated maintenance scheduling


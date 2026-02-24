# Deep Learning Turbulence Prediction System

## Overview

This project is an enhanced version of the original Cloud-Based Turbulence Prediction System.  
The initial version used a Random Forest model on satellite-derived atmospheric parameters.  
This upgraded version focuses on integrating Deep Learning techniques and improved atmospheric feature modeling for more robust turbulence prediction.

The system aims to predict aviation turbulence severity using satellite data and atmospheric variables, with future scope for real-time automation and intelligent advisory generation.

---

## Problem Statement

Clear Air Turbulence (CAT) is difficult to detect using traditional radar systems because it occurs in cloud-free regions.  

This project explores the use of satellite-derived atmospheric parameters and machine learning models to predict turbulence severity in advance.

---

## Objectives

- Improve turbulence prediction using Deep Learning models
- Move from heuristic severity classification to physically grounded modeling (e.g., EDR-based thresholds)
- Incorporate additional atmospheric variables such as wind shear and stability indices
- Develop a scalable and partially automated prediction pipeline
- Explore intelligent advisory generation for aviation use

---

## System Architecture (Planned V2)

1. Satellite Data Ingestion (INSAT-3D/3DR HDF5)
2. Data Cleaning and Feature Extraction
3. Extended Atmospheric Feature Engineering
4. Deep Learning Model (CNN / CNN-LSTM)
5. Severity Estimation (Regression + Classification)
6. API Layer (Flask/FastAPI)
7. Visualization Dashboard
8. Future: Automated Retraining & Advisory Layer

---

## Key Enhancements Over V1

- Transition from Random Forest to Deep Learning
- Addition of spatio-temporal modeling
- Expanded atmospheric feature set
- Improved evaluation methodology
- Modular architecture for future scalability

---

## Technologies Used

- Python
- NumPy, Pandas
- TensorFlow / PyTorch (planned)
- Scikit-learn
- HDF5 data processing
- Flask / FastAPI
- Docker (for containerization)

---

## Evaluation Strategy

The system will be evaluated using:

- Confusion Matrix
- Precision, Recall, F1 Score
- ROC-AUC
- RMSE / MAE (for regression models)
- Cross-validation techniques

Future versions aim to validate predictions against turbulence severity metrics such as Eddy Dissipation Rate (EDR).

---

## Repository Structure (Planned)

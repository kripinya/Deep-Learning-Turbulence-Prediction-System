# 🛰️ Turbulence Insight: Cloud-Based Aviation Safety System
### Advanced Turbulence Prediction using Satellite Data Ingestion and Machine Learning

![Dashboard Preview](https://img.shields.io/badge/Status-Operational-brightgreen)
![Tech Stack](https://img.shields.io/badge/Stack-Flask%20|%20RF%20|%20Docker%20|%20HDF5-blue)

---

## 🔍 1. The Research Gap & Motivation

Turbulence remains one of the leading causes of non-fatal injuries in commercial aviation and significant operational costs due to rerouting and structural maintenance. 

**Traditional methods face three primary gaps:**
1.  **Vertical Resolution Gap**: Standard meteorological models often lack the fine-grained vertical resolution (e.g., at the 100m vs 10m levels) necessary to calculate precise wind shear in real-time.
2.  **Ingestion Bottleneck**: Raw satellite data from providers like **MOSDAC (INSAT-3D/3DR)** is delivered in complex HDF5 formats. Most existing systems require asynchronous, offline processing before any predictive analysis can occur.
3.  **Visualization Gap**: There is a lack of accessible, unified dashboards that can simultaneously handle single-point analysis, bulk historical verification, and real-time satellite streaming.

**Turbulence Insight** bridges these gaps by providing an end-to-end pipeline that converts raw satellite telemetry into actionable aviation risk intelligence using high-performance Machine Learning.

---

## 🛠️ 2. Core Functionalities

The system features a state-of-the-art **Interactive Dashboard** (Glassmorphism design) with four specialized analytical pathways:

### 🚄 A. Instant Predict
*   **Purpose**: Real-time risk assessment for a specific flight coordinate.
*   **Function**: Accepts atmospheric inputs (Temperature, Dewpoint, Surface Pressure, Wind Speeds) and returns a categorical risk level (**Low, Moderate, Severe**) with a confidence score.

### 📊 B. Batch CSV Processing
*   **Purpose**: Bulk analysis for post-flight verification or regional mapping.
*   **Function**: Processes datasets containing thousands of records instantly. It provides a **Global Risk Profile** logic, calculating the percentage distribution of turbulence risks across the entire dataset.

### 📐 C. Raw HDF5 Data Converter & Analyzer
*   **Purpose**: Native ingestion of satellite-grade telemetry.
*   **Function**: Bypasses manual preprocessing by allowing users to upload `.h5` files directly. The system extracts geospatial metadata, flattens the arrays into ML-ready formats, and generates an immediate **Turbulence Intensity Profile**.

### 📡 D. Live MOSDAC Ingestion (MOSDAC-X)
*   **Purpose**: Real-time forecasting and situational awareness.
*   **Function**: Connects to the MOSDAC API skeleton to fetch live product metadata. It features a **scrolling data feed** and a **24-hour predictive forecast** trend, projecting future risks based on current atmospheric trends.

---

## 🧠 3. Technical Implementation & Features

### Machine Learning Engine
*   **Model**: Random Forest Classifier trained on expanded meteorological datasets.
*   **Feature Engineering**: The system automatically derives critical indicators:
    *   **Wind Shear**: Calculated as the absolute difference between wind speeds at 100m and 10m.
    *   **Dewpoint Depression (Dewpt Dep.)**: The difference between Temperature and Dewpoint, a key indicator of atmospheric stability and moisture-driven vertical movement.
*   **Inference Pipeline**: Built to handle both numerical and categorical outputs with a dynamic label mapper.

### The Backend Architecture
*   **API**: Flask-based RESTful service optimized for high-concurrency with Gunicorn.
*   **Compatibility**: Includes a unique shim for **Python 3.14+** support, handling standard library attribute changes (`pkgutil.get_loader`).
*   **Containerization**: Fully Dockerized for seamless movement between local development and Cloud (AWS) environments.

---

## 🚀 4. Setup & Installation

### Option A: Using Docker (Recommended)
```bash
# Build the image
docker build -t turbulence-api .

# Run the container (Access at http://localhost:8080)
docker run -d -p 8080:8080 --name turbulence-api-container turbulence-api
```

### Option B: Manual Setup
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Model Artifacts**:
    ```bash
    python3 train_model.py
    ```
3.  **Start Server**:
    ```bash
    gunicorn -w 4 -b 0.0.0.0:8080 api.app:app
    ```

---

## 📈 5. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict` | POST | Single point prediction. |
| `/predict-batch` | POST | Bulk CSV prediction + Global Risk Profile. |
| `/process-h5` | POST | Raw HDF5 conversion + Severity Analysis. |
| `/mosdac-ingest` | POST | Continuous live satellite data ingestion. |
| `/health` | GET | System health and model availability check. |

---

## 🗺️ 6. Future Roadmap
*   **AWS Deployment**: Migration to ECS with Auto-scaling and S3-based artifact storage.
*   **Dynamic GIS Overlay**: Integrating mapping libraries to visualize results over geographic flight paths.
*   **Deep Learning (LSTM)**: Incorporating temporal sequences for improved forecasting Accuracy.

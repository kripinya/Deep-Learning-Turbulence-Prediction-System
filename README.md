# Cloud-Based-Turbulence-Prediction-System-Aviation-Safety-

Developing a real-time turbulence detection system using INSAT-3D/3DR satellite data. Built a data ingestion and preprocessing pipeline, generated cleaned feature datasets, and designed scalable cloud infrastructure for model deployment. Worked with AWS services, HDF5 processing, ML pipelines, and REST API integration.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training (Optional)
If you need to train a fresh model using Open-Meteo ERA5 data:
```bash
python3 train_model.py --lat 28.6 --lon 77.2 --start 2024-01-01 --end 2024-01-31
```
This will save `rf_model.joblib` and `scaler.joblib` to `model_artifacts/`.

### 3. Running the Inference API
```bash
# Set model path (if different from default)
export MODEL_PATH="model_artifacts/model_rf_pipeline.joblib"
export SCALER_PATH="model_artifacts/scaler.joblib"

# Start the server (Dev)
python3 api/app.py

# Start the server (Prod)
gunicorn -w 2 -b 0.0.0.0:8080 api.app:app
```

### 4. Testing the Endpoint
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"temperature_2m": 25, "dewpoint_2m": 20, "surface_pressure": 1013, "wind_speed_10m": 5, "wind_speed_100m": 12, "relative_humidity_2m": 70, "cloud_cover": 50, "lat": 28.6, "lon": 77.2}]}'
```

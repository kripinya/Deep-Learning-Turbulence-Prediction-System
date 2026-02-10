# app.py
import os
import io
import logging
from typing import List, Dict
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- Py3.14 Compatibility Patch ---
import pkgutil
import importlib.util
import sys

if not hasattr(pkgutil, 'get_loader'):
    def get_loader(name):
        try:
            spec = importlib.util.find_spec(name)
            return spec.loader if spec else None
        except (ImportError, AttributeError, ValueError):
            return None
    pkgutil.get_loader = get_loader
# ----------------------------------

from flask import render_template

import joblib
import pandas as pd
import numpy as np
import h5py
from datetime import datetime, timedelta
try:
    from api.mosdac_client import MosdacClient
except ImportError:
    from mosdac_client import MosdacClient

# --- config (update if you prefer S3) ---
MODEL_PATH = os.getenv("MODEL_PATH", "model_artifacts/rf_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "model_artifacts/scaler.joblib")
PORT = int(os.getenv("PORT", 8080))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("turbulence-api")

app = Flask(__name__)

# Load model (joblib pipeline expected)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    logger.info(f"Loading model from {path}")
    m = joblib.load(path)
    logger.info("Model loaded")
    return m

try:
    MODEL = load_model(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        logger.info(f"Loading scaler from {SCALER_PATH}")
        SCALER = joblib.load(SCALER_PATH)
    else:
        logger.warning(f"Scaler not found at {SCALER_PATH}")
        SCALER = None
except Exception as e:
    logger.exception("Failed to load model or scaler on startup")
    MODEL = None
    SCALER = None

# Features exptected by the model
EXPECTED_FEATURES = ["wind_speed_10m", "wind_speed_100m", "wind_shear", "relative_humidity_2m", "cloud_cover", "surface_pressure", "dewpt_dep"]

# Helpful label map - change if your labels differ
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "Severe"}

def df_from_request(req) -> pd.DataFrame:
    """Accept JSON array of records or file upload (CSV or gzipped CSV) or form fields."""
    ct = (req.content_type or "").lower()
    # JSON body
    if "application/json" in ct:
        data = req.get_json(force=True)
        if isinstance(data, dict):
            # allow {"rows": [ {...}, ... ]} or single record
            if "rows" in data and isinstance(data["rows"], list):
                data = data["rows"]
            else:
                data = [data]
        return pd.DataFrame(data)
    # File upload (form-data)
    if 'file' in req.files:
        f = req.files['file']
        filename = secure_filename(f.filename or "upload.csv")
        raw = f.read()
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ""
        if ext in ("gz", "gzip"):
            return pd.read_csv(io.BytesIO(raw), compression='gzip')
        else:
            return pd.read_csv(io.BytesIO(raw))
    # Form fields -> single-row
    if req.form:
        d = {k: req.form.get(k) for k in req.form.keys()}
        return pd.DataFrame([d])
    raise ValueError("Unsupported input. Send JSON array or upload a CSV file (field 'file').")

def calculate_risk_summary(predictions: List[Dict]) -> Dict:
    """Helper to calculate percentage distribution of risk levels."""
    total = len(predictions)
    counts = {"Low": 0, "Moderate": 0, "Severe": 0}
    for p in predictions:
        label = p.get("pred_text")
        if label in counts:
            counts[label] += 1
    
    return {
        label: round((count / total) * 100, 1) if total > 0 else 0
        for label, count in counts.items()
    }

def ensure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    If model expects lat_bin/lon_bin but they are missing, infer from lat/lon.
    Strategy: lat_bin = int(lat), lon_bin = int(lon) when lat/lon present.
    """
    # only attempt if lat/lon present
    if 'lat_bin' not in df.columns and 'lat' in df.columns:
        logger.info("Auto-filling missing column: lat_bin from lat (int(lat))")
        df['lat_bin'] = df['lat'].apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    if 'lon_bin' not in df.columns and 'lon' in df.columns:
        logger.info("Auto-filling missing column: lon_bin from lon (int(lon))")
        df['lon_bin'] = df['lon'].apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    return df

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    ok = MODEL is not None
    return jsonify({"status":"ok" if ok else "model_missing", "model_path": MODEL_PATH, "scaler_loaded": SCALER is not None}), (200 if ok else 500)

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Endpoint for uploading a CSV and getting batch predictions with summary."""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        df = df_from_request(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    results, status = predict_internal(df)
    if status != 200:
        return results, status
    
    predictions = results.get_json()["results"]
    summary = calculate_risk_summary(predictions)

    return jsonify({
        "total_records": len(predictions),
        "risk_summary": summary,
        "results": predictions[:100] # return first 100 for preview
    }), 200

def predict_internal(df: pd.DataFrame):
    """Refactored core prediction logic for reuse."""
    # Convert numeric-like columns to numeric
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception: pass

    df = ensure_bins(df)
    
    # Feature Engineering
    if 'wind_shear' not in df.columns and 'wind_speed_100m' in df.columns and 'wind_speed_10m' in df.columns:
        df['wind_shear'] = (df['wind_speed_100m'] - df['wind_speed_10m']).abs()
    if 'dewpt_dep' not in df.columns and 'temperature_2m' in df.columns and 'dewpoint_2m' in df.columns:
        df['dewpt_dep'] = df['temperature_2m'] - df['dewpoint_2m']

    # Get feature order
    feature_names = None
    if hasattr(MODEL, "feature_names_in_"):
        feature_names = list(MODEL.feature_names_in_)
    
    # Ensure all required columns exist in the dataframe before slicing
    target_cols = feature_names if feature_names else EXPECTED_FEATURES
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_for_pred = df[target_cols].copy()

    X_for_pred_scaled = SCALER.transform(X_for_pred) if SCALER else X_for_pred

    try:
        preds = MODEL.predict(X_for_pred_scaled)
        probs = MODEL.predict_proba(X_for_pred_scaled) if hasattr(MODEL, "predict_proba") else None
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    out = []
    for i, p in enumerate(preds):
        # If prediction is already a string (like 'Moderate'), use it directly
        if isinstance(p, (str, np.str_)):
            label = str(p)
        else:
            try:
                label = LABEL_MAP.get(int(p), str(p))
            except (ValueError, TypeError):
                label = str(p)
        
        rec = {"index": i, "pred_text": label}
        if probs is not None:
            rec["probs"] = [float(x) for x in probs[i]]
        out.append(rec)

    return jsonify({"results": out}), 200

@app.route("/process-h5", methods=["POST"])
def process_h5():
    """Convert uploaded HDF5 to processed CSV format."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    f = request.files['file']
    filename = secure_filename(f.filename)
    # Save temporarily
    tmp_path = os.path.join("/tmp", filename)
    f.save(tmp_path)

    try:
        with h5py.File(tmp_path, "r") as h5:
            lat_ds = h5.get("Latitude") or h5.get("CSBT_Latitude")
            lon_ds = h5.get("Longitude") or h5.get("CSBT_Longitude")
            if lat_ds is None:
                return jsonify({"error": "No geospatial data found in H5"}), 400
            
            lat = np.array(lat_ds).ravel()
            lon = np.array(lon_ds).ravel()
            ctp = h5.get("CTP")
            ctt = h5.get("CTT")

            # Simple masking for demo (using logic from process_mosdac_perfile.py)
            mask = (lat != 32767)
            df = pd.DataFrame({
                "lat": lat[mask],
                "lon": lon[mask],
                "CTP": np.array(ctp).ravel()[mask] if ctp else np.nan,
                "CTT": np.array(ctt).ravel()[mask] if ctt else np.nan
            })
            
            # Prediction Integration
            pred_df = df.copy()
            pred_df['cloud_cover'] = pred_df['CTP'].fillna(0) / 10 # dummy mapping
            pred_df['surface_pressure'] = 1013 # default
            pred_json, status = predict_internal(pred_df)
            predictions = pred_json.get_json()["results"]

            # Calculate Aggregate Risk Summary
            risk_summary = calculate_risk_summary(predictions)

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            return jsonify({
                "message": "H5 processed and analyzed (Global Summary)",
                "rows": len(predictions),
                "risk_summary": risk_summary,
                "predictions_preview": predictions[:10], # Keep a small preview
                "csv_preview": csv_buf.getvalue()[:2000]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route("/mosdac-ingest", methods=["POST"])
def mosdac_ingest():
    """Live MOSDAC ingestion trigger."""
    data = request.get_json() or {}
    client = MosdacClient(data.get("username"), data.get("password"))
    
    lat = data.get("lat", 28.6)
    lon = data.get("lon", 77.2)
    dataset = data.get("dataset", "3D_IMG_L2B_CTP")

    logger.info(f"Triggering MOSDAC ingestion for {dataset} at {lat}, {lon}")
    result = client.get_realtime_data(dataset, lat, lon)
    
    if "error" in result:
        return jsonify(result), 401
    
    # Calculate prediction for the LATEST point in the stream
    latest_point = result["stream"][-1]
    mock_row = pd.DataFrame([{
        "lat": lat, "lon": lon,
        "temperature_2m": 25, "dewpoint_2m": 20, # defaults
        "surface_pressure": 1013, 
        "wind_speed_10m": 5, "wind_speed_100m": 12,
        "relative_humidity_2m": 70, 
        "cloud_cover": latest_point["CTP"] / 10 # Map CTP to cloud cover
    }])
    pred_res, status = predict_internal(mock_row)
    
    return jsonify({
        "mosdac_status": "Live Streaming Active",
        "ingestion_info": result,
        "current_prediction": pred_res.get_json()["results"][0]
    }), status

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded on server."}), 500
    try:
        df = df_from_request(request)
    except Exception as e:
        return jsonify({"error": f"Could not parse input: {e}"}), 400

    # Keep original index mapping
    original_index = df.index.tolist()

    # Convert numeric-like columns to numeric (best-effort)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass

    # Auto-create lat_bin/lon_bin if missing
    df = ensure_bins(df)

    # --- Feature Engineering ---
    # Calculate derived features if missing
    if 'wind_shear' not in df.columns and 'wind_speed_100m' in df.columns and 'wind_speed_10m' in df.columns:
        df['wind_shear'] = (df['wind_speed_100m'] - df['wind_speed_10m']).abs()
    
    if 'dewpt_dep' not in df.columns and 'temperature_2m' in df.columns and 'dewpoint_2m' in df.columns:
        df['dewpt_dep'] = df['temperature_2m'] - df['dewpoint_2m']
    # ---------------------------

    # Try to obtain the trained feature order (feature_names_in_)
    feature_names = None
    try:
        # direct attribute on estimator/pipeline
        if hasattr(MODEL, "feature_names_in_"):
            feature_names = list(MODEL.feature_names_in_)
        else:
            # if it's a Pipeline, search steps for the first estimator that recorded feature names
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(MODEL, Pipeline):
                    for name, step in MODEL.named_steps.items():
                        if hasattr(step, "feature_names_in_"):
                            feature_names = list(step.feature_names_in_)
                            break
            except Exception:
                feature_names = None
    except Exception:
        feature_names = None

    # If we found a canonical feature order, reindex the dataframe to match it (fill missing cols with NaN)
    X_for_pred = df
    if feature_names:
        logger.info(f"Reindexing input to model feature order: {feature_names}")
        # ensure all feature names exist as columns (add with NaN if missing), then order them
        for col in feature_names:
            if col not in df.columns:
                df[col] = pd.NA
        X_for_pred = df[feature_names].copy()
    else:
        # Fallback: use hardcoded expected features
        logger.info(f"No feature_names_in_ found on MODEL; using hardcoded list: {EXPECTED_FEATURES}")
        # Ensure all columns exist
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0.0 # Fill missing with 0 or suitable default
        X_for_pred = df[EXPECTED_FEATURES].copy()

    # Apply scaling if available
    if SCALER:
        try:
            logger.info("Applying feature scaling")
            X_for_pred_scaled = SCALER.transform(X_for_pred)
        except Exception as e:
            logger.warning(f"Scaling failed: {e}. Proceeding without scaling.")
            X_for_pred_scaled = X_for_pred
    else:
        X_for_pred_scaled = X_for_pred

    # Attempt prediction
    try:
        preds = MODEL.predict(X_for_pred_scaled)
        probs = MODEL.predict_proba(X_for_pred_scaled) if hasattr(MODEL, "predict_proba") else None
    except Exception as e:
        logger.exception("Primary prediction attempt failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    out = []
    for i, p in enumerate(preds):
        # Handle string or int predictions
        if isinstance(p, (int, np.integer, float, np.floating)):
            pred_label = int(p)
            pred_text = LABEL_MAP.get(pred_label, str(pred_label))
        else:
            pred_label = str(p)
            pred_text = pred_label

        rec = {"index": original_index[i], "pred_label": pred_label, "pred_text": pred_text}
        if probs is not None:
            rec["probs"] = [float(x) for x in probs[i]]
        out.append(rec)

    return jsonify({"n_rows": len(out), "results": out}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

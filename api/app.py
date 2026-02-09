# app.py
import os
import io
import logging
from typing import List, Dict
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import joblib
import pandas as pd
import numpy as np

# --- config (update if you prefer S3) ---
MODEL_PATH = os.getenv("MODEL_PATH", "model_artifacts/model_rf_pipeline.joblib")
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

@app.route("/health", methods=["GET"])
def health():
    ok = MODEL is not None
    return jsonify({"status":"ok" if ok else "model_missing", "model_path": MODEL_PATH}), (200 if ok else 500)

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

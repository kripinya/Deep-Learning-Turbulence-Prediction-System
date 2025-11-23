# predict.py
import sys
import pandas as pd
from joblib import load
import numpy as np
import os

SCALER = "model_artifacts/scaler.joblib"
MODEL  = "model_artifacts/rf_model.joblib"

def get_expected_features(scaler):
    # If scaler was fitted on a DataFrame, it often has feature_names_in_
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # fallback to common default (six features used earlier)
    return ["wind_speed_10m","wind_speed_100m","wind_shear",
            "relative_humidity_2m","cloud_cover","surface_pressure"]

def predict_dataframe(df):
    scaler = load(SCALER)
    model  = load(MODEL)
    features = get_expected_features(scaler)

    # ensure DataFrame contains all expected features
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    X = df[features].values
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)
    return preds, probs, features

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        preds, probs, features = predict_dataframe(df)
        out = pd.DataFrame(df[features].reset_index(drop=True))
        out["pred"] = preds
        out["proba_max"] = probs.max(axis=1)
        out.to_csv("predictions.csv", index=False)
        print("Saved predictions.csv")
    else:
        # quick demo single sample: build a zeroed sample for all expected features
        from joblib import load
        scaler = load(SCALER)
        feat = get_expected_features(scaler)
        demo_dict = {f: 0.0 for f in feat}
        # Replace some demo values (optional)
        if "wind_speed_10m" in demo_dict:
            demo_dict["wind_speed_10m"] = 5.2
        if "wind_speed_100m" in demo_dict:
            demo_dict["wind_speed_100m"] = 12.1
        if "wind_shear" in demo_dict:
            demo_dict["wind_shear"] = abs(demo_dict["wind_speed_100m"] - demo_dict["wind_speed_10m"])
        demo = pd.DataFrame([demo_dict])
        p, prob, features = predict_dataframe(demo)
        print("Demo pred:", p[0], "prob:", prob[0].max())

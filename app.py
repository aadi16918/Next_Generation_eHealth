
import os
import json
from pathlib import Path
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import psycopg
import shap

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

load_dotenv(BASE_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATASET_PATH = BASE_DIR / "dataset" / "synthetic_IoMT_70.csv"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/frontend",
)
CORS(app)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["feature_columns"]
REQUIRES_SCALING = metadata.get("requires_scaling", False)
scaler = (
    joblib.load(SCALER_PATH)
    if REQUIRES_SCALING
    else None
)


def prepare_input(data):
    if not isinstance(data, dict):
        raise ValueError("Send measurements as a JSON object.")

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        raise ValueError(f"Missing measurements: {', '.join(missing)}")

    values = {}
    for feature in FEATURES:
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be a number.")

        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")

        values[feature] = value

    frame = pd.DataFrame([values], columns=FEATURES)
    transformed = scaler.transform(frame) if REQUIRES_SCALING else frame
    return frame, transformed


def get_probabilities(model_input):
    result = {"class_0_probability": None, "class_1_probability": None}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        for label, probability in zip(model.classes_, probabilities):
            if int(label) == 0:
                result["class_0_probability"] = float(probability)
            elif int(label) == 1:
                result["class_1_probability"] = float(probability)

    return result


def json_safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def error_response(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# Website pages
@app.get("/")
@app.get("/dashboard")
def dashboard():
    return app.send_static_file("index.html")


@app.get("/monitoring")
def monitoring_page():
    return app.send_static_file("monitoring.html")


@app.get("/prediction")
def prediction_page():
    return app.send_static_file("prediction.html")


@app.get("/history")
def history_page():
    return app.send_static_file("history.html")


# Backend status
@app.get("/api/health")
def health_check():
    database_status = "not_configured"

    if DATABASE_URL:
        try:
            with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            database_status = "connected"
        except Exception:
            app.logger.exception("Database health check failed")
            database_status = "error"

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "database": database_status,
    })


# Latest readings and saved AI predictions
@app.get("/api/readings")
def get_readings():
    if not DATABASE_URL:
        return error_response("DATABASE_URL is not configured.", 500)

    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)

    limit = max(1, min(limit or 100, 1000))
    offset = max(0, offset or 0)

    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.patient_readings"
                )
                total = cur.fetchone()[0]

                cur.execute(
                    """
                    SELECT reading_id, created_at, ecg, bioimpedence,
                           speed, distance, spo2, predicted_class,
                           class_0_probability, class_1_probability
                    FROM public.patient_readings
                    ORDER BY created_at DESC NULLS LAST,
                             reading_id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )

                columns = [column.name for column in cur.description]
                rows = [
                    dict(zip(columns, row))
                    for row in cur.fetchall()
                ]

        rows = [
            {key: json_safe(value) for key, value in row.items()}
            for row in rows
        ]

        latest = rows[0] if rows else None

        return jsonify({
            "status": "success",
            "total": total,
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "latest": latest,
            "readings": rows,
        })

    except Exception:
        app.logger.exception("Failed to fetch readings")
        return error_response("Could not fetch patient readings.", 500)


# Run a new AI prediction from submitted measurements
@app.post("/api/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        measurements, model_input = prepare_input(data)

        predicted_class = int(model.predict(model_input)[0])
        probabilities = get_probabilities(model_input)

        return jsonify({
            "status": "success",
            "predicted_class": predicted_class,
            "prediction": predicted_class,
            "measurements": measurements.iloc[0].to_dict(),
            **probabilities,
        })

    except ValueError as error:
        return error_response(str(error))
    except Exception:
        app.logger.exception("Prediction failed")
        return error_response("AI prediction failed.", 500)


# Explain the prediction using SHAP
@app.post("/api/explain")
def explain():
    try:
        data = request.get_json(silent=True) or {}
        measurements, model_input = prepare_input(data)

        predicted_class = int(model.predict(model_input)[0])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)

        if isinstance(shap_values, list):
            class_index = list(model.classes_).index(predicted_class)
            values = np.asarray(shap_values[class_index]).reshape(-1)
        else:
            values = np.asarray(shap_values)

            # Some SHAP versions return:
            # samples x features x classes
            if values.ndim == 3:
                class_index = list(model.classes_).index(predicted_class)
                values = values[0, :, class_index]
            else:
                values = values.reshape(-1)

        contributions = [
            {"feature": feature, "shap_value": float(value)}
            for feature, value in zip(FEATURES, values)
        ]
        contributions.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return jsonify({
            "status": "success",
            "predicted_class": predicted_class,
            "explanations": contributions,
            "features": contributions,
        })

    except ValueError as error:
        return error_response(str(error))
    except Exception:
        app.logger.exception("SHAP explanation failed")
        return error_response("Could not generate SHAP explanation.", 500)


# Basic counterfactual: search dataset rows with a different predicted class
@app.post("/api/counterfactual")
def counterfactual():
    try:
        data = request.get_json(silent=True) or {}
        measurements, model_input = prepare_input(data)

        current_class = int(model.predict(model_input)[0])
        target_class = data.get("target_class", 1 - current_class)

        try:
            target_class = int(target_class)
        except (TypeError, ValueError):
            return error_response("target_class must be an integer.")

        if not DATASET_PATH.exists():
            return error_response("Counterfactual dataset was not found.", 500)

        dataset = pd.read_csv(DATASET_PATH)
        if not all(feature in dataset.columns for feature in FEATURES):
            return error_response(
                "Dataset does not contain all required model features.", 500
            )

        dataset = dataset[FEATURES].apply(pd.to_numeric, errors="coerce")
        dataset = dataset.dropna()

        if dataset.empty:
            return error_response("No usable dataset rows were found.", 500)

        candidates = dataset.copy()
        candidate_input = (
            scaler.transform(candidates)
            if REQUIRES_SCALING
            else candidates
        )
        predictions = model.predict(candidate_input)
        candidates = candidates.loc[
            np.asarray(predictions) == target_class
        ].copy()

        if candidates.empty:
            return error_response(
                f"No example found for target class {target_class}.", 404
            )

        # Return the closest dataset example by standardized distance.
        reference = measurements.iloc[0].astype(float)
        scale = dataset.std().replace(0, 1).fillna(1)
        distances = ((candidates - reference) / scale).pow(2).sum(axis=1)
        best_index = distances.idxmin()
        alternative = candidates.loc[best_index].to_dict()

        changes = {
            feature: {
                "current": float(reference[feature]),
                "suggested": float(alternative[feature]),
            }
            for feature in FEATURES
        }

        return jsonify({
            "status": "success",
            "current_class": current_class,
            "target_class": target_class,
            "counterfactual": alternative,
            "changes": changes,
            "note": (
                "Experimental dataset-based example only; "
                "not a clinical recommendation."
            ),
        })

    except ValueError as error:
        return error_response(str(error))
    except Exception:
        app.logger.exception("Counterfactual generation failed")
        return error_response("Could not generate counterfactual.", 500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )

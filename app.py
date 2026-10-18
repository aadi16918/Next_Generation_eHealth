
import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
import shap
import psycopg

from datetime import date, datetime
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from flask_cors import CORS


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "synthetic_IoMT_70.csv"
)

from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="/frontend"
)
CORS(app)


# ==================================================
# 2. LOAD MODEL AND METADATA
# ==================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

FEATURES = metadata["feature_columns"]
REQUIRES_SCALING = metadata.get("requires_scaling", False)


def prepare_input(data):
    """Apply scaling only when the selected model requires it."""
    frame = pd.DataFrame(data, columns=FEATURES)

    if REQUIRES_SCALING:
        return scaler.transform(frame)

    return frame


def parse_measurements(data):
    """Validate the five required input measurements."""
    if not isinstance(data, dict):
        raise ValueError("Send measurements as a JSON object.")

    values = []

    for feature in FEATURES:
        if feature not in data:
            raise KeyError(feature)

        values.append(float(data[feature]))

    if not np.all(np.isfinite(values)):
        raise ValueError("Measurements must be finite numbers.")

    return pd.DataFrame([values], columns=FEATURES)


def get_class_probabilities(model_input):
    """Return probabilities if supported by the selected model."""
    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(model_input)[0]

    return {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }


def error_response(message, status_code=400):
    return jsonify({
        "status": "error",
        "message": message
    }), status_code


# ==================================================
# 3. FRONTEND PAGES
# ==================================================

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


# ==================================================
# 4. API HOME
# ==================================================

@app.get("/api")
def api_home():
    return jsonify({
        "status": "success",
        "message": "Next Generation Healthcare API is running!"
    })


# ==================================================
# 5. DATABASE HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health_check():
    if not DATABASE_URL:
        return error_response("DATABASE_URL is missing.", 500)

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

        return jsonify({
            "status": "success",
            "database": "connected"
        })

    except Exception:
        app.logger.exception("Database health check failed")
        return error_response("Database connection failed.", 500)


# ==================================================
# 6. FETCH PATIENT READINGS
# ==================================================


@app.get("/api/readings")
def get_readings():
    if not DATABASE_URL:
        return error_response("DATABASE_URL is missing.", 500)

    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)

    limit = max(1, min(limit or 100, 1000))
    offset = max(0, offset or 0)

    try:
        with psycopg.connect(DATABASE_URL) as conn:
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
                    ORDER BY created_at DESC NULLS LAST, reading_id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )

                columns = [column.name for column in cur.description]
                rows = [
                    dict(zip(columns, row))
                    for row in cur.fetchall()
                ]

        for row in rows:
            for key, value in row.items():
                if isinstance(value, (date, datetime)):
                    row[key] = value.isoformat()

        return jsonify({
            "status": "success",
            "total": total,
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "readings": rows
        })

    except Exception:
        app.logger.exception("Failed to fetch patient readings")
        return error_response("Could not fetch patient readings.", 500)

# ==================================================
# 7. LIST API ROUTES
# ==================================================

@app.get("/api/routes")
def list_routes():
    routes = []

    for rule in app.url_map.iter_rules():
        methods = sorted(
            method for method in rule.methods
            if method not in ("HEAD", "OPTIONS")
        )

        routes.append({
            "path": rule.rule,
            "methods": methods
        })

    return jsonify({"routes": routes})


# ==================================================
# 8. AI PREDICTION
# ==================================================

@app.post("/api/predict")
def predict():
    try:
        original_df = parse_measurements(
            request.get_json(silent=True)
        )

        model_input = prepare_input(original_df)
        prediction = int(model.predict(model_input)[0])

        return jsonify({
            "status": "success",
            "predicted_class": prediction,
            "class_probabilities": get_class_probabilities(model_input),
            "notice": (
                "Experimental prediction from a synthetic-data model; "
                "not a medical diagnosis."
            )
        })

    except KeyError as exc:
        return error_response(
            f"Missing measurement: {exc.args[0]}"
        )

    except (TypeError, ValueError) as exc:
        return error_response(str(exc))

    except Exception:
        app.logger.exception("Prediction failed")
        return error_response("Prediction failed.", 500)


# ==================================================
# 9. SHAP EXPLANATION
# ==================================================

@app.post("/api/explain")
def explain_prediction():
    try:
        original_df = parse_measurements(
            request.get_json(silent=True)
        )

        model_input = prepare_input(original_df)
        prediction = int(model.predict(model_input)[0])

        # TreeExplainer supports tree-based models such as
        # Decision Tree and Random Forest.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)

        if isinstance(shap_values, list):
            class_index = list(model.classes_).index(prediction)
            contributions = np.asarray(
                shap_values[class_index]
            )[0]

        else:
            contributions = np.asarray(shap_values)

            if contributions.ndim == 3:
                class_index = list(model.classes_).index(prediction)
                contributions = contributions[0, :, class_index]

            elif contributions.ndim == 2:
                contributions = contributions[0]

            else:
                contributions = contributions.reshape(-1)

        explanation = {
            feature: float(value)
            for feature, value in zip(FEATURES, contributions)
        }

        return jsonify({
            "status": "success",
            "predicted_class": prediction,
            "shap_values": explanation,
            "notice": (
                "Experimental explanation using synthetic data; "
                "not a medical diagnosis."
            )
        })

    except KeyError as exc:
        return error_response(
            f"Missing measurement: {exc.args[0]}"
        )

    except (TypeError, ValueError) as exc:
        return error_response(str(exc))

    except Exception:
        app.logger.exception("SHAP explanation failed")
        return error_response(
            "Could not generate explanation. Check model compatibility.",
            500
        )


# ==================================================
# 10. COUNTERFACTUAL EXPLANATION
# ==================================================

@app.post("/api/counterfactual")
def counterfactual_prediction():
    try:
        original_df = parse_measurements(
            request.get_json(silent=True)
        )

        original_input = prepare_input(original_df)
        original_class = int(model.predict(original_input)[0])
        target_class = 1 - original_class

        if not os.path.exists(DATASET_PATH):
            return error_response("Counterfactual dataset not found.", 500)

        dataset = pd.read_csv(DATASET_PATH)
        X = dataset[FEATURES]

        if X.isnull().any().any():
            return error_response(
                "Counterfactual dataset contains missing values.", 500
            )

        rng = np.random.default_rng(42)
        lower = X.quantile(0.05).to_numpy()
        upper = X.quantile(0.95).to_numpy()

        candidates = pd.DataFrame(
            rng.uniform(
                lower,
                upper,
                size=(20000, len(FEATURES))
            ),
            columns=FEATURES
        )

        candidate_input = prepare_input(candidates)
        candidate_predictions = model.predict(candidate_input)

        candidates = candidates.loc[
            candidate_predictions == target_class
        ].copy()

        if candidates.empty:
            return error_response(
                "No counterfactual candidate was found.", 404
            )

        # Standardize only for distance calculation, not prediction.
        distance_scaler = StandardScaler().fit(X)

        original_distance = distance_scaler.transform(original_df)
        candidate_distances = distance_scaler.transform(candidates)

        distances = np.sqrt(
            ((candidate_distances - original_distance) ** 2).sum(axis=1)
        )

        best_index = int(np.argmin(distances))
        counterfactual = candidates.iloc[best_index]

        counterfactual_df = pd.DataFrame(
            [counterfactual],
            columns=FEATURES
        )

        counterfactual_input = prepare_input(counterfactual_df)
        counterfactual_class = int(
            model.predict(counterfactual_input)[0]
        )

        if counterfactual_class != target_class:
            raise RuntimeError("Counterfactual verification failed.")

        changes = {
            feature: {
                "original": float(original_df.iloc[0][feature]),
                "counterfactual": float(counterfactual[feature]),
                "change": float(
                    counterfactual[feature]
                    - original_df.iloc[0][feature]
                )
            }
            for feature in FEATURES
        }

        return jsonify({
            "status": "success",
            "original_prediction": original_class,
            "original_class_probabilities": get_class_probabilities(
                original_input
            ),
            "counterfactual_prediction": counterfactual_class,
            "counterfactual_class_probabilities": get_class_probabilities(
                counterfactual_input
            ),
            "counterfactual_values": {
                feature: float(counterfactual[feature])
                for feature in FEATURES
            },
            "feature_changes": changes,
            "standardized_distance": float(distances[best_index]),
            "notice": (
                "Experimental counterfactual from synthetic data; "
                "not medical advice."
            )
        })

    except KeyError as exc:
        return error_response(
            f"Missing measurement: {exc.args[0]}"
        )

    except (TypeError, ValueError) as exc:
        return error_response(str(exc))

    except Exception:
        app.logger.exception("Counterfactual generation failed")
        return error_response(
            "Counterfactual generation failed.", 500
        )


# ==================================================
# 11. RUN FLASK
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)


import os
import time
import uuid
import json
import random
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["feature_columns"]
REQUIRES_SCALING = metadata.get("requires_scaling", False)
scaler = joblib.load(SCALER_PATH) if REQUIRES_SCALING else None

REQUIRED_FEATURES = {
    "ecg", "bioimpedence", "speed", "distance", "spo2"
}
if not REQUIRED_FEATURES.issubset(FEATURES):
    raise ValueError(
        f"Model features don't match simulator measurements. "
        f"Expected {sorted(REQUIRED_FEATURES)}, got {FEATURES}"
    )


def generate_reading():
    return {
        "ecg": round(random.uniform(60, 160), 2),
        "bioimpedence": round(random.uniform(10, 35), 2),
        "speed": round(random.uniform(0, 8), 2),
        "distance": round(random.uniform(0, 10), 2),
        "spo2": round(random.uniform(85, 100), 2),
    }


def predict_reading(measurements):
    frame = pd.DataFrame([measurements], columns=FEATURES)

    model_input = (
        scaler.transform(frame)
        if REQUIRES_SCALING
        else frame
    )

    prediction = int(model.predict(model_input)[0])
    p0, p1 = None, None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]

        for label, probability in zip(model.classes_, probabilities):
            if int(label) == 0:
                p0 = float(probability)
            elif int(label) == 1:
                p1 = float(probability)

    return prediction, p0, p1


def save_reading(reading_id, measurements, prediction, p0, p1):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.patient_readings (
                    reading_id, created_at, ecg, bioimpedence,
                    speed, distance, spo2, predicted_class,
                    class_0_probability, class_1_probability,
                    data_source, processing_location
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                (
                    reading_id,
                    datetime.now(timezone.utc),
                    measurements["ecg"],
                    measurements["bioimpedence"],
                    measurements["speed"],
                    measurements["distance"],
                    measurements["spo2"],
                    prediction,
                    p0,
                    p1,
                    "laptop_simulator",
                    "local",
                ),
            )


def main():
    print("IoMT simulator started.")
    print("Saving a synthetic reading every 5 seconds.")
    print("Press Ctrl+C to stop.")

    while True:
        reading_id = f"realtime-{uuid.uuid4().hex[:12]}"

        try:
            measurements = generate_reading()
            prediction, p0, p1 = predict_reading(measurements)
            save_reading(reading_id, measurements, prediction, p0, p1)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{reading_id} | "
                f"ECG={measurements['ecg']} | "
                f"SpO2={measurements['spo2']} | "
                f"Prediction={prediction} | "
                f"Class 0={p0} | Class 1={p1}"
            )

        except KeyboardInterrupt:
            print("\nSimulator stopped.")
            break
        except Exception as error:
            print(f"Reading failed: {error}")

        time.sleep(5)


if __name__ == "__main__":
    main()

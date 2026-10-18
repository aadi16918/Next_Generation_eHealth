
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "synthetic_IoMT_70.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

OUTPUT_PATH = ROOT / "results" / "simulated_patient_readings.csv"

# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# ==========================================
# 3. LOAD MODEL AND METADATA
# ==========================================

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["feature_columns"]

# ==========================================
# 4. LOAD ALL EXISTING READINGS
# ==========================================

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Synthetic dataset is empty.")

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing feature columns: {missing}")

readings = df[features].apply(pd.to_numeric, errors="coerce")

if readings.isnull().any().any():
    raise ValueError("Dataset contains missing or invalid feature values.")

if not np.isfinite(readings.to_numpy(dtype=float)).all():
    raise ValueError("Dataset contains non-finite values.")

# ==========================================
# 5. RUN AI INFERENCE IN BATCHES
# ==========================================

batch_size = 5000
prediction_parts = []
probability_parts = []

scaler = None

if metadata.get("requires_scaling", False):
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

for start in range(0, len(readings), batch_size):
    batch = readings.iloc[start:start + batch_size]

    model_input = scaler.transform(batch) if scaler is not None else batch

    prediction_parts.append(
        model.predict(model_input).astype(int)
    )

    probability_parts.append(
        model.predict_proba(model_input)
    )

    print(f"Processed {min(start + batch_size, len(readings))}/{len(readings)} readings")

predictions = np.concatenate(prediction_parts)
probabilities = np.vstack(probability_parts)

# ==========================================
# 6. PREPARE SIMULATED RECORDS
# ==========================================

output = readings.copy()

output.insert(
    0,
    "reading_id",
    [f"synthetic-{i + 1}" for i in range(len(output))]
)

output.insert(
    1,
    "timestamp",
    pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),
        periods=len(output),
        freq="s"
    ).astype(str)
)

output["predicted_class"] = predictions
output["class_0_probability"] = probabilities[:, 0]
output["class_1_probability"] = probabilities[:, 1]
output["data_source"] = "synthetic_simulation"

# ==========================================
# 7. SAVE ALL READINGS
# ==========================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n===== IoMT SIMULATION COMPLETE =====")
print("Source records:", len(df))
print("Readings generated:", len(output))
print("Saved to:", OUTPUT_PATH)
print("\nPrediction counts:")
print(output["predicted_class"].value_counts().sort_index())

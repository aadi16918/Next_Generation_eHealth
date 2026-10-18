
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "results" / "simulated_patient_readings.csv"
OUTPUT_PATH = ROOT / "results" / "edge_processed_readings.csv"

MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

BATCH_SIZE = 5000

# ==========================================
# 2. LOAD FILES
# ==========================================

for path in [INPUT_PATH, MODEL_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise ValueError("No simulated readings found.")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

model = joblib.load(MODEL_PATH)
features = metadata["feature_columns"]

missing = [name for name in features if name not in df.columns]

if missing:
    raise ValueError(f"Missing feature columns: {missing}")

# ==========================================
# 3. VALIDATE ALL READINGS
# ==========================================

input_df = df[features].apply(pd.to_numeric, errors="coerce")

if input_df.isnull().any().any():
    raise ValueError("Input contains missing or non-numeric values.")

if not np.isfinite(input_df.to_numpy(dtype=float)).all():
    raise ValueError("Input contains invalid numeric values.")

# ==========================================
# 4. LOAD SCALER IF REQUIRED
# ==========================================

scaler = None

if metadata.get("requires_scaling", False):
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

# ==========================================
# 5. PROCESS EVERY READING
# ==========================================

predictions = []
probability_batches = []

for start in range(0, len(input_df), BATCH_SIZE):
    batch = input_df.iloc[start:start + BATCH_SIZE]

    model_input = scaler.transform(batch) if scaler is not None else batch

    predictions.append(model.predict(model_input).astype(int))
    probability_batches.append(model.predict_proba(model_input))

    print(
        f"Edge processed {min(start + BATCH_SIZE, len(input_df))}"
        f"/{len(input_df)} readings"
    )

prediction = np.concatenate(predictions)
probabilities = np.vstack(probability_batches)

# ==========================================
# 6. BUILD OUTPUT
# ==========================================

output = df.copy()

# Preserve source timestamps and reading IDs.
output["predicted_class"] = prediction
output["class_0_probability"] = probabilities[:, 0]
output["class_1_probability"] = probabilities[:, 1]
output["processing_location"] = "local_edge"
output["data_source"] = "synthetic_simulation"

# ==========================================
# 7. SAVE OUTPUT
# ==========================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("\n===== EDGE PROCESSING COMPLETE =====")
print("Input readings:", len(df))
print("Processed readings:", len(output))
print("Saved to:", OUTPUT_PATH)
print("\nPrediction counts:")
print(output["predicted_class"].value_counts().sort_index())

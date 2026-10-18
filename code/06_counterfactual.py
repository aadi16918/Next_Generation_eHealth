
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Define project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "synthetic_IoMT_70.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)
METADATA_PATH = os.path.join(
    BASE_DIR, "models", "model_metadata.json"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "xai")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_columns"]
REQUIRES_SCALING = metadata.get("requires_scaling", False)

# 3. Load dataset and model
df = pd.read_csv(DATA_PATH)

if "heart_attack" not in df.columns:
    raise ValueError("Target column 'heart_attack' not found.")

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

if X.columns.tolist() != FEATURES:
    raise ValueError("Dataset features do not match model metadata.")

if df.isnull().any().any():
    raise ValueError("Dataset contains missing values.")

model = joblib.load(MODEL_PATH)

# 4. Prepare model prediction function
def predict_classes(data):
    data = pd.DataFrame(data, columns=FEATURES)

    if REQUIRES_SCALING:
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        scaler = joblib.load(scaler_path)
        data = scaler.transform(data)

    return model.predict(data)


def predict_probabilities(data):
    data = pd.DataFrame(data, columns=FEATURES)

    if REQUIRES_SCALING:
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        scaler = joblib.load(scaler_path)
        data = scaler.transform(data)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(data)

    return None


# 5. Select original sample
original_index = 0
original = X.iloc[original_index].copy()
original_df = pd.DataFrame([original], columns=FEATURES)

original_prediction = int(predict_classes(original_df)[0])
original_probabilities = predict_probabilities(original_df)

original_probability = (
    float(original_probabilities[0, 1])
    if original_probabilities is not None
    else None
)

target_class = 1 - original_prediction

print("\n===== ORIGINAL INSTANCE =====")
print(original.to_string())
print("Original prediction:", original_prediction)
print("Probability of class 1:", original_probability)
print("Desired counterfactual class:", target_class)

# 6. Generate candidates within observed feature ranges
rng = np.random.default_rng(42)
number_of_candidates = 20000

lower_bounds = X.quantile(0.05)
upper_bounds = X.quantile(0.95)

candidates = pd.DataFrame(
    rng.uniform(
        low=lower_bounds.to_numpy(),
        high=upper_bounds.to_numpy(),
        size=(number_of_candidates, len(FEATURES))
    ),
    columns=FEATURES
)

candidate_predictions = predict_classes(candidates)
valid_mask = candidate_predictions == target_class
valid_candidates = candidates.loc[valid_mask].copy()

print("\nCandidates generated:", len(candidates))
print("Candidates reaching target class:", len(valid_candidates))

# 7. Find the nearest candidate
if valid_candidates.empty:
    raise RuntimeError(
        "No counterfactual found. Try increasing the candidate count "
        "or inspect the dataset and model."
    )

distance_scaler = StandardScaler()
distance_scaler.fit(X)

original_distance = distance_scaler.transform(original_df)
valid_distance = distance_scaler.transform(valid_candidates)

distances = np.sqrt(
    np.sum((valid_distance - original_distance) ** 2, axis=1)
)

best_position = int(np.argmin(distances))
counterfactual = valid_candidates.iloc[best_position].copy()
counterfactual_df = pd.DataFrame(
    [counterfactual], columns=FEATURES
)

# 8. Verify counterfactual prediction
counterfactual_prediction = int(
    predict_classes(counterfactual_df)[0]
)

counterfactual_probabilities = predict_probabilities(
    counterfactual_df
)

counterfactual_probability = (
    float(counterfactual_probabilities[0, 1])
    if counterfactual_probabilities is not None
    else None
)

if counterfactual_prediction != target_class:
    raise RuntimeError("Counterfactual verification failed.")

# 9. Calculate feature changes
feature_changes = pd.DataFrame({
    "Feature": FEATURES,
    "Original": original.to_numpy(dtype=float),
    "Counterfactual": counterfactual.to_numpy(dtype=float)
})

feature_changes["Change"] = (
    feature_changes["Counterfactual"]
    - feature_changes["Original"]
)
feature_changes["Absolute_Change"] = (
    feature_changes["Change"].abs()
)

print("\n===== COUNTERFACTUAL EXPLANATION =====")
print("\nCounterfactual values:")
print(counterfactual.to_string())

print("\nOriginal class:", original_prediction)
print("Counterfactual class:", counterfactual_prediction)
print("Original class-1 probability:", original_probability)
print("Counterfactual class-1 probability:", counterfactual_probability)
print("Standardized distance:", round(float(distances[best_position]), 4))

print("\n===== FEATURE CHANGES =====")
print(feature_changes.to_string(index=False))

# 10. Save results
result = counterfactual.to_frame().T
result["original_prediction"] = original_prediction
result["counterfactual_prediction"] = counterfactual_prediction
result["original_probability"] = original_probability
result["counterfactual_probability"] = counterfactual_probability
result["standardized_distance"] = float(distances[best_position])

result.to_csv(
    os.path.join(OUTPUT_DIR, "counterfactual_result.csv"),
    index=False
)

feature_changes.to_csv(
    os.path.join(OUTPUT_DIR, "counterfactual_feature_changes.csv"),
    index=False
)

print("\n===== COUNTERFACTUAL COMPLETE =====")
print("Saved: xai/counterfactual_result.csv")
print("Saved: xai/counterfactual_feature_changes.csv")
print(
    "Note: This explains model behavior using synthetic data; "
    "it is not medical advice."
)

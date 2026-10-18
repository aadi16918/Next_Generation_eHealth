
import os
import json
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# 1. Load dataset
df = pd.read_csv("dataset/synthetic_IoMT_70.csv")

if "heart_attack" not in df.columns:
    raise ValueError("Target column 'heart_attack' not found.")

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# 2. Load trained model
model = joblib.load("models/best_model.pkl")

# 3. Validate feature order using model metadata
metadata_path = "models/model_metadata.json"

if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if X.columns.tolist() != metadata["feature_columns"]:
        raise ValueError(
            "Dataset feature order does not match model metadata."
        )

    requires_scaling = metadata["requires_scaling"]
else:
    raise FileNotFoundError(
        "models/model_metadata.json not found. "
        "Run code/03_train_models.py first."
    )

# 4. Prepare data for the model
scaler = joblib.load("models/scaler.pkl")

def prepare_data(data):
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != X.shape[1]:
        raise ValueError("Incorrect number of input features.")

    if requires_scaling:
        return scaler.transform(data)

    return data

# 5. Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["No Heart Attack", "Heart Attack"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# 6. Select one sample
sample_index = 0
sample = X.iloc[sample_index].to_numpy(dtype=float)

# 7. Prediction function for LIME
def predict_proba(data):
    prepared = prepare_data(data)
    probabilities = model.predict_proba(prepared)

    if probabilities.shape[1] != 2:
        raise ValueError("Expected a binary classification model.")

    return probabilities

# 8. Generate explanation
explanation = explainer.explain_instance(
    data_row=sample,
    predict_fn=predict_proba,
    labels=(0, 1),
    num_features=len(X.columns),
    num_samples=3000
)

# 9. Save explanation
os.makedirs("xai", exist_ok=True)

output_path = "xai/lime_explanation.html"
explanation.save_to_file(output_path)

# 10. Display results
prepared_sample = prepare_data(sample)
prediction = int(model.predict(prepared_sample)[0])
probabilities = predict_proba(sample)[0]

print("\n===== LIME EXPLANATION COMPLETE =====")
print("Model:", type(model).__name__)
print("Selected sample:", sample_index)

print("\nFeature values:")
for feature, value in zip(X.columns, sample):
    print(f"{feature}: {value:.4f}")

print("\nPredicted class:", prediction)
print(
    "Predicted label:",
    "Heart Attack" if prediction == 1 else "No Heart Attack"
)

print("\nPrediction probabilities:")
print(f"No Heart Attack: {probabilities[0]:.4f}")
print(f"Heart Attack:    {probabilities[1]:.4f}")

print("\nLIME contributions for each class:")
for label, label_name in [(0, "No Heart Attack"), (1, "Heart Attack")]:
    print(f"\n{label_name}:")
    for feature, contribution in explanation.as_list(label=label):
        print(f"  {feature}: {contribution:.4f}")

print("\nExplanation saved to:", output_path)


import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("dataset/synthetic_IoMT_70.csv")

if "heart_attack" not in df.columns:
    raise ValueError("Target column 'heart_attack' not found.")

X = df.drop(columns=["heart_attack"])

# 2. Load model and metadata
model = joblib.load("models/best_model.pkl")

metadata_path = "models/model_metadata.json"
if os.path.exists(metadata_path):
    import json

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected_features = metadata["feature_columns"]

    if list(X.columns) != expected_features:
        raise ValueError(
            "Dataset feature columns do not match model metadata."
        )

# 3. Select a reproducible sample
X_sample = X.sample(
    n=min(1000, len(X)),
    random_state=42
)

# 4. Create SHAP explainer
explainer = shap.TreeExplainer(model)
raw_shap = explainer(X_sample)

# 5. Extract SHAP values across SHAP versions
if isinstance(raw_shap, list):
    shap_values = raw_shap[1]
elif hasattr(raw_shap, "values"):
    shap_values = raw_shap.values
else:
    shap_values = raw_shap

# Select class 1 when output includes both binary classes
if isinstance(shap_values, list):
    shap_values = shap_values[1]

if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

if shap_values.shape != X_sample.shape:
    raise ValueError(
        f"Unexpected SHAP shape {shap_values.shape}; "
        f"expected {X_sample.shape}."
    )

# 6. Save summary plot
os.makedirs("xai", exist_ok=True)

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=X_sample.columns.tolist(),
    plot_type="dot",
    show=False
)

plt.title("SHAP Summary - Decision Tree")
plt.tight_layout()

plt.savefig(
    "xai/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 7. Print results
print("===== SHAP EXPLANATION COMPLETE =====")
print("Model:", type(model).__name__)
print("Samples explained:", len(X_sample))
print("Features explained:", X_sample.shape[1])
print("SHAP values shape:", shap_values.shape)
print("Features:", X_sample.columns.tolist())
print("Plot saved to: xai/shap_summary.png")

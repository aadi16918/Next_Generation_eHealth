
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load and validate dataset
df = pd.read_csv("dataset/synthetic_IoMT_70.csv")

if "heart_attack" not in df.columns:
    raise ValueError("Target column 'heart_attack' not found.")

if df.isnull().any().any():
    raise ValueError("Dataset contains missing values. Clean them first.")

X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected binary target labels 0 and 1.")

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "SVM": LinearSVC(
        max_iter=5000, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

scaled_models = {"Logistic Regression", "SVM", "KNN"}

results = []
trained_models = {}
reports = {}

print("\n===== MODEL TRAINING STARTED =====\n")

for name, model in models.items():
    print(f"Training: {name}")

    if name in scaled_models:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    trained_models[name] = model

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }
    results.append(metrics)

    reports[name] = {
        "classification_report": classification_report(
            y_test, predictions,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1 Score : {metrics['F1_Score']:.4f}\n")

# 5. Save comparison and detailed evaluation
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

with open("results/model_evaluation.json", "w") as f:
    json.dump(reports, f, indent=2)

print("===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# 6. Select best model by F1 score
best_model_name = results_df.loc[
    results_df["F1_Score"].idxmax(), "Model"
]
best_model = trained_models[best_model_name]

# 7. Save model, scaler and preprocessing metadata
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

metadata = {
    "model_name": best_model_name,
    "target_column": "heart_attack",
    "feature_columns": X.columns.tolist(),
    "requires_scaling": best_model_name in scaled_models,
    "selection_metric": "F1_Score",
    "random_state": 42,
    "test_size": 0.2
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nBest model:", best_model_name)
print("Saved: models/best_model.pkl")
print("Saved: models/scaler.pkl")
print("Saved: models/model_metadata.json")
print("Saved: results/model_comparison.csv")
print("Saved: results/model_evaluation.json")
print("\n===== TRAINING COMPLETE =====")

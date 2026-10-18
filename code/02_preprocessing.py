import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/synthetic_IoMT_70.csv")

# Separate features and target
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display results
print("===== PREPROCESSING COMPLETE =====")
print("Original dataset:", df.shape)
print("Training features:", X_train_scaled.shape)
print("Testing features:", X_test_scaled.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
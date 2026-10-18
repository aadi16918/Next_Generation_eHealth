import pandas as pd

# Load dataset
df = pd.read_csv("dataset/synthetic_IoMT_70.csv")

# Basic information
print("===== DATASET INFORMATION =====")
print("Shape:", df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(df["heart_attack"].value_counts())

print("\n===== FIRST 5 ROWS =====")
print(df.head())
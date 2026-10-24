# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 4: FEATURE ENGINEERING AND DATA PREPARATION
# ============================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("output/results", exist_ok=True)

# Step 31: Load cleaned dataset
clean_data_path = "data/student_productivity_cleaned.csv"
if os.path.exists(clean_data_path):
    df = pd.read_csv(clean_data_path)
else:
    df = pd.read_csv("C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv")

# Step 32: Create target variable
def assign_productivity_level(score):
    if score < 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"

df["Productivity_Level"] = df["productivity_score"].apply(assign_productivity_level)
print("\nProductivity Level Counts:")
print(df["Productivity_Level"].value_counts())

# Save target distribution
df["Productivity_Level"].value_counts().to_csv("output/results/productivity_level_distribution.csv")

# Step 33: Separate features and target
X = df.drop(["student_id", "productivity_score", "final_grade", "Productivity_Level"], axis=1, errors='ignore')
y = df["Productivity_Level"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Step 34: Encode categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)
print("\nShape after encoding:", X.shape)
print(X.head())

# Step 35: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\nTraining target distribution:\n", y_train.value_counts())
print("\nTesting target distribution:\n", y_test.value_counts())

# Step 36: Standard Scaling for Logistic Regression and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)
print("\nDay 4 preprocessing completed.")

# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 2: DATA CLEANING AND PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import os

# Load dataset
dataset_path = "data/student_productivity_distraction_dataset_20000.csv"
if not os.path.exists(dataset_path):
    dataset_path = "C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv"

df = pd.read_csv(dataset_path)

# Step 13: Create a copy of original dataset
df_clean = df.copy()

print("\nOriginal dataset shape:")
print(df_clean.shape)

# Step 14: Check duplicate records
print("\nDuplicates before cleaning:")
print(df_clean.duplicated().sum())

# Step 15: Remove duplicate records
df_clean = df_clean.drop_duplicates()

print("\nDuplicates after cleaning:")
print(df_clean.duplicated().sum())

# Step 16: Check missing values
print("\nMissing values before cleaning:")
print(df_clean.isnull().sum())

# Step 17: Handle missing values
# Fill numerical missing values with median
numeric_columns = df_clean.select_dtypes(include=np.number).columns
for col in numeric_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Fill categorical missing values with mode
categorical_columns = df_clean.select_dtypes(include="object").columns
for col in categorical_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Step 18: Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df_clean.to_csv("data/student_productivity_cleaned.csv", index=False)
print("\nCleaned dataset saved to data/student_productivity_cleaned.csv")
print("Day 2 data cleaning and preprocessing completed.")

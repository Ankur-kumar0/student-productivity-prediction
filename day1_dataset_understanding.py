# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 1: DATASET UNDERSTANDING
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 2: Load the dataset
dataset_path = "data/student_productivity_distraction_dataset_20000.csv"
if not os.path.exists(dataset_path):
    dataset_path = "C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv"

df = pd.read_csv(dataset_path)

# Step 3: Display first five rows
print("First five rows:")
print(df.head())

# Step 4: Check dataset shape
print("\nDataset shape:")
print(df.shape)

# Step 5: Display column names
print("\nColumn names:")
print(df.columns)

# Step 6: Check data types
print("\nDataset information:")
df.info()

# Step 7: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 8: Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Step 9: Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Step 10: Check gender distribution
print("\nGender distribution:")
print(df["gender"].value_counts())

# Step 11: Check productivity score
print("\nProductivity score distribution:")
print(df["productivity_score"].describe())

# Step 12: Check final grade
print("\nFinal grade distribution:")
print(df["final_grade"].describe())

print("\nDay 1 dataset understanding completed.")

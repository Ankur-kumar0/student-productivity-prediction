# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Step 23: Load cleaned dataset
clean_data_path = "data/student_productivity_cleaned.csv"
if os.path.exists(clean_data_path):
    df_clean = pd.read_csv(clean_data_path)
else:
    df_clean = pd.read_csv("C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv")

print("\nCleaned dataset:")
print(df_clean.head())

# Step 24: Gender distribution
print("\nOBJECTIVE 1: GENDER ANALYSIS")
gender_count = df_clean["gender"].value_counts()
print(gender_count)
gender_count.to_csv("output/results/gender_analysis.csv")

plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="gender")
plt.title("Student Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("output/figures/gender_analysis.png")
plt.close()

# Step 27: Phone usage analysis
print("\nOBJECTIVE 4: PHONE USAGE ANALYSIS")
print(df_clean["phone_usage_hours"].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, y="phone_usage_hours")
plt.title("Student Phone Usage")
plt.ylabel("Phone Usage Hours")
plt.tight_layout()
plt.savefig("output/figures/phone_usage.png")
plt.close()

# Step 29: Correlation analysis
print("\nOBJECTIVE 6: CORRELATION ANALYSIS")
correlation = df_clean.select_dtypes(include=np.number).corr()
print(correlation)

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Student Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/figures/correlation_heatmap.png")
plt.close()

# Step 30: Final EDA summary
print("\n========== DAY 3 SUMMARY ==========")
print("Total cleaned records:", len(df_clean))
print("Total visualizations created: 3")
print("Charts saved in output/figures/")
print("Analysis completed successfully.")

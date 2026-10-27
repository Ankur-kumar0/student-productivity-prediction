# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION (END-TO-END PIPELINE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# ------------------------------------------------------------
# DAY 1: DATASET UNDERSTANDING
# ------------------------------------------------------------
dataset_path = "data/student_productivity_distraction_dataset_20000.csv"
if not os.path.exists(dataset_path):
    dataset_path = "C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv"

df = pd.read_csv(dataset_path)
print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ------------------------------------------------------------
# DAY 2: DATA CLEANING AND PREPROCESSING
# ------------------------------------------------------------
df_clean = df.copy().drop_duplicates()
numeric_columns = df_clean.select_dtypes(include=np.number).columns
for col in numeric_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

categorical_columns = df_clean.select_dtypes(include="object").columns
for col in categorical_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

df_clean.to_csv("data/student_productivity_cleaned.csv", index=False)

# ------------------------------------------------------------
# DAY 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
gender_count = df_clean["gender"].value_counts()
gender_count.to_csv("output/results/gender_analysis.csv")

plt.figure(figsize=(7, 5))
sns.countplot(data=df_clean, x="gender")
plt.title("Student Gender Distribution")
plt.savefig("output/figures/gender_analysis.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, y="phone_usage_hours")
plt.title("Student Phone Usage")
plt.savefig("output/figures/phone_usage.png")
plt.close()

correlation = df_clean.select_dtypes(include=np.number).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Student Dataset Correlation Heatmap")
plt.savefig("output/figures/correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# DAY 4: FEATURE ENGINEERING AND DATA PREPARATION
# ------------------------------------------------------------
def assign_productivity_level(score):
    if score < 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"

df_clean["Productivity_Level"] = df_clean["productivity_score"].apply(assign_productivity_level)
df_clean["Productivity_Level"].value_counts().to_csv("output/results/productivity_level_distribution.csv")

X = df_clean.drop(["student_id", "productivity_score", "final_grade", "Productivity_Level"], axis=1, errors='ignore')
y = df_clean["Productivity_Level"]
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# DAY 5: IMPLEMENTATION OF MACHINE LEARNING MODELS
# ------------------------------------------------------------
logistic_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)

nb_model = GaussianNB().fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# ------------------------------------------------------------
# DAY 6: MODEL COMPARISON AND BEST MODEL SELECTION
# ------------------------------------------------------------
model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Gaussian Naive Bayes", "KNN", "Decision Tree"],
    "Accuracy": [accuracy_score(y_test, y_pred_logistic), accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_dt)],
    "Precision": [precision_score(y_test, y_pred_logistic, average="macro", zero_division=0), precision_score(y_test, y_pred_nb, average="macro", zero_division=0), precision_score(y_test, y_pred_knn, average="macro", zero_division=0), precision_score(y_test, y_pred_dt, average="macro", zero_division=0)],
    "Recall": [recall_score(y_test, y_pred_logistic, average="macro", zero_division=0), recall_score(y_test, y_pred_nb, average="macro", zero_division=0), recall_score(y_test, y_pred_knn, average="macro", zero_division=0), recall_score(y_test, y_pred_dt, average="macro", zero_division=0)],
    "F1_Score": [f1_score(y_test, y_pred_logistic, average="macro", zero_division=0), f1_score(y_test, y_pred_nb, average="macro", zero_division=0), f1_score(y_test, y_pred_knn, average="macro", zero_division=0), f1_score(y_test, y_pred_dt, average="macro", zero_division=0)]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(model_comparison.round(4))
model_comparison.to_csv("output/results/model_comparison.csv", index=False)

best_model = model_comparison.loc[model_comparison["Accuracy"].idxmax()]
print("\nBEST MODEL:", best_model["Model"], "with Accuracy:", round(best_model["Accuracy"], 4))

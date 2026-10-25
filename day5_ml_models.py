# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 5: IMPLEMENTATION OF MACHINE LEARNING MODELS
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
    classification_report
)

os.makedirs("output/figures", exist_ok=True)

# Prepare dataset
clean_data_path = "data/student_productivity_cleaned.csv"
if os.path.exists(clean_data_path):
    df = pd.read_csv(clean_data_path)
else:
    df = pd.read_csv("C:/Users/ak737/Downloads/archive (3)/student_productivity_distraction_dataset_20000.csv")

def assign_productivity_level(score):
    if score < 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"

df["Productivity_Level"] = df["productivity_score"].apply(assign_productivity_level)
X = df.drop(["student_id", "productivity_score", "final_grade", "Productivity_Level"], axis=1, errors='ignore')
y = df["Productivity_Level"]
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 38: Logistic Regression Model
print("\n--- 1. LOGISTIC REGRESSION ---")
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic, labels=["Low", "Medium", "High"]))
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Classification Report:\n", classification_report(y_test, y_pred_logistic, zero_division=0))

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_logistic, labels=["Low", "Medium", "High"]),
            annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/figures/logistic_regression_cm.png")
plt.close()

# Step 39: Gaussian Naive Bayes Model
print("\n--- 2. GAUSSIAN NAIVE BAYES ---")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb, labels=["Low", "Medium", "High"]))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb, zero_division=0))

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_nb, labels=["Low", "Medium", "High"]),
            annot=True, fmt="d", cmap="Greens", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/figures/naive_bayes_cm.png")
plt.close()

# Step 40: K-Nearest Neighbors Model
print("\n--- 3. K-NEAREST NEIGHBORS ---")
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn, labels=["Low", "Medium", "High"]))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn, labels=["Low", "Medium", "High"]),
            annot=True, fmt="d", cmap="Oranges", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/figures/knn_cm.png")
plt.close()

# Step 41: Decision Tree Model
print("\n--- 4. DECISION TREE ---")
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt, labels=["Low", "Medium", "High"]))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))

plt.figure(figsize=(24, 12))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=["High", "Low", "Medium"], rounded=True, max_depth=3, fontsize=9)
plt.title("Student Productivity Decision Tree")
plt.tight_layout()
plt.savefig("output/figures/decision_tree.png", dpi=300, bbox_inches="tight")
plt.close()

# Step 42: Day 5 Summary
print("\n========== DAY 5 SUMMARY ==========")
print("Models implemented: Logistic Regression, GaussianNB, KNN, Decision Tree")
print("Evaluation: Accuracy, Precision, Recall, F1-Score")
print("Day 5 machine learning completed.")

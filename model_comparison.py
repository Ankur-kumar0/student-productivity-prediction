# ============================================================
# STUDENT PRODUCTIVITY LEVEL PREDICTION
# DAY 6: MODEL COMPARISON AND BEST MODEL SELECTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("output/results", exist_ok=True)
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

# Train models
logistic_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)

nb_model = GaussianNB().fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Step 43: Model comparison
model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Gaussian Naive Bayes", "KNN", "Decision Tree"],
    "Accuracy": [accuracy_score(y_test, y_pred_logistic), accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_dt)],
    "Precision": [precision_score(y_test, y_pred_logistic, average="macro", zero_division=0), precision_score(y_test, y_pred_nb, average="macro", zero_division=0), precision_score(y_test, y_pred_knn, average="macro", zero_division=0), precision_score(y_test, y_pred_dt, average="macro", zero_division=0)],
    "Recall": [recall_score(y_test, y_pred_logistic, average="macro", zero_division=0), recall_score(y_test, y_pred_nb, average="macro", zero_division=0), recall_score(y_test, y_pred_knn, average="macro", zero_division=0), recall_score(y_test, y_pred_dt, average="macro", zero_division=0)],
    "F1_Score": [f1_score(y_test, y_pred_logistic, average="macro", zero_division=0), f1_score(y_test, y_pred_nb, average="macro", zero_division=0), f1_score(y_test, y_pred_knn, average="macro", zero_division=0), f1_score(y_test, y_pred_dt, average="macro", zero_division=0)]
})

print("\n========== MODEL COMPARISON ==========")
print(model_comparison.round(4))
model_comparison.to_csv("output/results/model_comparison.csv", index=False)

# Step 44: Accuracy comparison graph
plt.figure(figsize=(10, 6))
plt.bar(model_comparison["Model"], model_comparison["Accuracy"], color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("output/figures/model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Step 45: Find the best model
best_model = model_comparison.loc[model_comparison["Accuracy"].idxmax()]
print("\n========== BEST MODEL ==========")
print("Best performing model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("F1 Score:", round(best_model["F1_Score"], 4))

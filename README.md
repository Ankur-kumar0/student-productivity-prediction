# 🎓 Student Productivity Level Prediction

A multi-class Machine Learning classification project predicting student productivity levels (**Low**, **Medium**, **High**) based on academic habits, phone distraction hours, and behavioral metrics.

---

## 📌 Project Overview
- **Domain:** Educational Data Mining / Machine Learning
- **Task:** 3-Class Classification (Low, Medium, High)
- **Implemented Models:** Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), Decision Tree
- **Evaluation Metrics:** Confusion Matrix, Accuracy, Precision, Recall, Macro F1-Score

---

## 📅 Daily Milestones & Commit History

| Day | Date | Commit Title | Description |
|---|---|---|---|
| **Day 1** | **Oct 21, 2026** | Add dataset understanding script and requirements | Initialized dataset loading, shape exploration, null checks, and stats. |
| **Day 2** | **Oct 22, 2026** | Implement data cleaning, duplicate removal and missing value imputation | Dropped duplicates, median imputation (numerical) & mode imputation (categorical). |
| **Day 3** | **Oct 23, 2026** | Add exploratory data analysis and correlation heatmaps | Gender distribution countplots, phone distraction boxplots, and correlation heatmap. |
| **Day 4** | **Oct 24, 2026** | Add target creation, feature encoding and standard scaling | Formed 3-tier target variable, one-hot encoded features, and applied StandardScaler. |
| **Day 5** | **Oct 25, 2026** | Implement ML classification models (LR, NB, KNN, DT) | Trained 4 classification models with confusion matrix visualizations & reports. |
| **Day 6** | **Oct 26, 2026** | Add model comparison metrics and best model selection | Benchmarked Accuracy, Precision, Recall, F1-scores, and plotted bar comparison. |
| **Day 7** | **Oct 27, 2026** | Add complete unified pipeline script and project documentation | Bundled end-to-end execution pipeline and completed technical documentation. |

---

## 📂 Project Structure

`	ext
├── data/
│   └── student_productivity_cleaned.csv  # Cleaned dataset
├── output/
│   ├── figures/                          # EDA charts, confusion matrices, comparison plots
│   └── results/                          # CSV metrics and evaluation tables
├── day1_dataset_understanding.py         # Day 1 exploration
├── day2_data_cleaning.py                 # Day 2 preprocessing
├── day3_eda.py                           # Day 3 visualizations
├── day4_feature_engineering.py           # Day 4 transformations
├── day5_ml_models.py                     # Day 5 model training
├── day6_model_comparison.py              # Day 6 comparison
├── student_productivity_pipeline.py      # Day 7 unified execution pipeline
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation
`

---

## ⚙️ Installation & Execution

1. **Clone the repository:**
   `ash
   git clone https://github.com/Ankur-kumar0/student-productivity-prediction.git
   cd student-productivity-prediction
   `

2. **Install requirements:**
   `ash
   pip install -r requirements.txt
   `

3. **Run the complete pipeline:**
   `ash
   python student_productivity_pipeline.py
   `

---

## 📊 Summary of Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | ~85% | 0.84 | 0.85 | 0.84 |
| **Gaussian Naive Bayes** | ~80% | 0.81 | 0.80 | 0.80 |
| **K-Nearest Neighbors (KNN)** | ~87% | 0.86 | 0.87 | 0.86 |
| **Decision Tree (Entropy)** | **~91%** | **0.91** | **0.91** | **0.91** |

---

## 👤 Author
- **Name:** Ankur Kumar
- **GitHub:** [@Ankur-kumar0](https://github.com/Ankur-kumar0)

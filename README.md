# 🩺 Diabetes Risk Classifier

A machine learning project that predicts the risk of diabetes based on medical measurements.

---

## 📌 Problem Statement

Diabetes is one of the most common chronic diseases worldwide.
The goal is to build a model that can predict whether a patient is at risk of diabetes
based on 8 medical features.

---

## 📊 Dataset

- **Source:** Pima Indians Diabetes Dataset (UCI)
- **Samples:** 768
- **Features:** 8 medical features
- **Classes:** No Diabetes (500) | Diabetes (268)

---

## 🛠️ Approach

| Step | Description |
|------|-------------|
| 1. EDA | Explored distributions, correlations, and outliers |
| 2. Data Cleaning | Replaced impossible zeros with median values |
| 3. Feature Engineering | Added BMI Category, Age Group, Glucose Level |
| 4. Preprocessing | StandardScaler + Train/Test Split |
| 5. Modeling | Logistic Regression, Random Forest, SVM, Gradient Boosting |
| 6. Tuning | GridSearchCV + Class Weight Balancing |
| 7. Deployment | Streamlit web application |

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 75.32% |
| ROC-AUC | 81.69% |
| Diabetes Recall | 80% |
| No Diabetes Precision | 87% |

**Best Model:** Logistic Regression with Class Balancing

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Tarekkareem123456/diabetes-classifier.git
cd diabetes-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧰 Technologies

| Library | Usage |
|---------|-------|
| Pandas | Data manipulation |
| Scikit-learn | Modeling & evaluation |
| Matplotlib/Seaborn | Visualization |
| Streamlit | Web deployment |
| Joblib | Model saving |
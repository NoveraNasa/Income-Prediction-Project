
````markdown
# 🧠 Income Prediction Project

This repository contains the solution for the **Intelligent Data Analysis** project.  
The goal is to **predict individual income groups (<=50K or >50K)** from demographic and employment data, and to assign predictions for the 25,000 individuals with unknown income levels.

---

## 📊 Dataset

- **Source:** `einkommen.train` (30,000 individuals)  
- **Labels available:** 5,000 individuals (<=50K, >50K)  
- **Unlabeled:** 25,000 individuals (`?`)  

**Features:**
- Age  
- Employment type  
- Weighting factor (sampling correction)  
- Education level & schooling years  
- Marital status  
- Employment area (occupation)  
- Partnership (relationship)  
- Ethnicity  
- Gender  
- Capital gains / losses  
- Weekly working time  
- Country of birth  
- **Income (target variable: <=50K, >50K, or ?)**  

---

## 🛠 Preprocessing

✔ Load data into **Pandas DataFrame**  
✔ Handle **missing values** (`?` → NaN or "Unknown")  
✔ **Categorical encoding** with One-Hot Encoding  
✔ **Normalization** of numerical attributes (Age, Hours per Week, Gains/Losses)  
✔ Split into:
- **Labeled set:** 5,000 rows with income known  
- **Unlabeled set:** 25,000 rows with income missing  

---

## 🤖 Methods

We compared baseline and stronger models:

- 🌿 **Logistic Regression (baseline):** linear classifier, interpretable coefficients, calibrated probabilities, fast training.  
- 🌳 **Random Forest (advanced):** ensemble of decision trees, captures non-linear interactions, provides feature importance.  

Both models used the same preprocessing pipeline and were tuned with **GridSearchCV** (LogReg: `C`; RF: `n_estimators`, `max_depth`, `min_samples_leaf`).

---

## 📈 Evaluation Protocol

- **Train/test split:** 80/20 stratified from the 5k labeled set  
- **Metrics:** Accuracy, Precision, Recall, **F1**, ROC-AUC  
- **Primary metric:** **F1-score**, chosen because the dataset is imbalanced and we care about balance between precision and recall for the >50K class.  
- **Cross-validation:** 5-fold CV during hyperparameter search  

---

## 📦 Outputs

- `outputs/labeled_5000.csv` → Cleaned labeled dataset  
- `outputs/unlabeled_25000.csv` → Cleaned unlabeled dataset  
- `outputs/predictions_25000.csv` → Predictions for the unlabeled individuals  
- `outputs/processed_known_features.csv` → Numeric feature matrix for correlation/regression  
- `outputs/processed_unknown_features.csv` → Numeric feature matrix for 25k unlabeled  
- `outputs/best_income_model.joblib` → Trained best model  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/NoveraNasa/Income-Prediction-Project.git
cd Income-Prediction-Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run training + predictions
python main.py

# (Optional) Run feature importance analysis
python feature_importance_rf.py
````

---

## 📊 Results

Below are the latest results evaluated on the held-out test set (20% of 5k labeled data).
Raw results also saved in [outputs/metrics.json](outputs/metrics.json).

<!-- RESULTS:START -->

### 📊 Results (auto-generated)

**Best model by F1:** `LogisticRegression` (F1 = 0.665)

| Model              | Accuracy | Precision | Recall |    F1 | ROC-AUC | Best Params                                              | CV F1 |
| ------------------ | -------: | --------: | -----: | ----: | ------: | -------------------------------------------------------- | ----: |
| LogisticRegression |    0.857 |     0.776 |  0.582 | 0.665 |   0.905 | `C`=1                                                    | 0.651 |
| RandomForest       |    0.849 |     0.746 |  0.578 | 0.651 |   0.904 | `max_depth`=20, `min_samples_leaf`=1, `n_estimators`=400 | 0.655 |

#### 🖼 Diagnostic Plots

Confusion matrices show error distribution (false positives/negatives).
ROC curves show separability across thresholds.

* **LogisticRegression**
  ![Confusion LogisticRegression](outputs/confusion_LogisticRegression.png)
  ![ROC LogisticRegression](outputs/roc_LogisticRegression.png)
* **RandomForest**
  ![Confusion RandomForest](outputs/confusion_RandomForest.png)
  ![ROC RandomForest](outputs/roc_RandomForest.png)

<!-- RESULTS:END -->

---

## 🏫 Academic Context

This project was completed as part of the **Intelligent Data Analysis** course at Universität Potsdam.
It demonstrates the full ML pipeline: problem setting, preprocessing, model comparison, evaluation, interpretability, and reproducibility.

---

## 📌 License

This project is released under the MIT License.
Feel free to use, modify, and share with proper attribution.

```

# 🧠 Income Prediction Project

This repository contains the solution for the **Intelligent Data Analysis**  project.  
The goal is to **predict individual income groups (<=50K or >50K)** from demographic and employment data, and to assign predictions for the 25,000 individuals with unknown income levels.

---
## Problem statement
Given feature vector x ∈ ℝ^d (categorical one-hot encoded + numeric scaled), learn f(x) = P(y=1|x) with y ∈ {0,1}, where y=1 ⇔ Income >50K. We optimize regularized empirical risk and select the best model using F1-score on a held-out test set. Finally, we apply f to the 25,000 unlabeled individuals (Income = “?”) to impute labels.

## 📊 Dataset

- **Source:** `einkommen.train` (30,000 individuals)  
- **Labels available:** 5,000 individuals (<=50K, >50K)  
- **Unlabeled:** 25,000 individuals (`?`)
- This data is taken from Uni-Potsdam repository 

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

## 🤖 Models

Logistic Regression: linear classifier, estimates P(y=1|x), uses sigmoid, interpretable weights, L2 regularization.

Random Forest: ensemble of trees, bootstrap sampling + random features, majority vote, handles nonlinearity + provides feature importance. 

---

## 📈 Evaluation

We used a stratified 80/20 train-test split to maintain class balance. F1-score was chosen as the primary metric because the dataset is imbalanced and we want a trade-off between precision and recall for the >50K class.

---

## 📦 Outputs

- `labeled_5000.csv` → Cleaned labeled dataset  
- `unlabeled_25000.csv` → Cleaned unlabeled dataset  
- `income_predictions.csv` → Predictions for the unlabeled individuals  
- Trained model saved with `joblib`  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/income-prediction.git
cd income-prediction

# Install dependencies
pip install -r requirements.txt

# Run the training + prediction pipeline
python main.py


📄 Documentation

The repository includes:

Jupyter Notebooks / Python scripts for preprocessing, training, and evaluation

Plots and tables: ROC curves, confusion matrices, feature importances

Project report explaining preprocessing decisions, model selection, and results

🏫 Academic Context

This project was completed as part of the Intelligent Data Analysis course/exam at Universität Potsdam, focusing on:

Data preprocessing and feature engineering

Model training & selection

Evaluation and interpretation

Applying predictions to unlabeled real-world data

📌 License

This project is released under the MIT License.
Feel free to use, modify, and share with proper attribution.

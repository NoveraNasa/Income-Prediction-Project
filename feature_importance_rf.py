# feature_importance_rf.py
"""
Random Forest Feature Importance Analysis
-----------------------------------------
This script loads the labeled 5000 dataset, trains a Random Forest,
and outputs the top features influencing income.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from main import (
    build_preprocessor, COLUMNS, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, prepare_labeled, load_and_split
)

# -----------------------
# 1) Load the labeled data
# -----------------------
labeled_df, unknown_df = load_and_split()
X_known, y_known = prepare_labeled(labeled_df)

# -----------------------
# 2) Train Random Forest
# -----------------------
rf_pipe = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ))
])
rf_pipe.fit(X_known, y_known)

# -----------------------
# 3) Extract feature names
# -----------------------
pre = rf_pipe.named_steps["preprocess"]
ohe = pre.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
all_features = np.r_[NUMERIC_FEATURES, cat_feature_names]

# -----------------------
# 4) Feature importance values
# -----------------------
importances = rf_pipe.named_steps["model"].feature_importances_
indices = np.argsort(importances)[::-1]

# -----------------------
# 5) Plot top 15
# -----------------------
top_n = 15
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices[:top_n]][::-1], align="center")
plt.yticks(range(top_n), [all_features[i] for i in indices[:top_n]][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features Influencing Income (Random Forest)")
plt.tight_layout()

# Save and show
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / "feature_importance_rf.png")
plt.show()

print(f"Feature importance plot saved → {OUTPUT_DIR/'feature_importance_rf.png'}")

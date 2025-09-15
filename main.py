# main.py
# Income Prediction Project — end-to-end training + prediction + exports

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from diagnostics import save_confusion_and_roc


# -----------------------
# 0) Configuration
# -----------------------
DATA_FILE = Path("einkommen.train")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMNS = [
    "Age","EmploymentType","WeightingFactor","EducationLevel","SchoolingYears",
    "MaritalStatus","EmploymentArea","Partnership","Ethnicity","Gender",
    "CapitalGains","CapitalLosses","WeeklyWorkingTime","CountryOfBirth","Income"
]

NUMERIC_FEATURES = [
    "Age", "WeightingFactor", "SchoolingYears",
    "CapitalGains", "CapitalLosses", "WeeklyWorkingTime"
]
CATEGORICAL_FEATURES = [
    "EmploymentType","EducationLevel","MaritalStatus","EmploymentArea",
    "Partnership","Ethnicity","Gender","CountryOfBirth"
]

# -----------------------
# utils
# -----------------------
def print_header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df

def build_preprocessor() -> ColumnTransformer:
    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, NUMERIC_FEATURES),
            ("cat", categorical_preprocess, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Works after .fit() is called (preprocessor is fitted inside Pipeline.fit()).
    """
    num_names = np.array(NUMERIC_FEATURES, dtype=object)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    return np.r_[num_names, cat_names]

def evaluate_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Try ROC-AUC
    roc = np.nan
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba)
    elif hasattr(model, "decision_function"):
        scores = pipe.decision_function(X_test)
        roc = roc_auc_score(y_test, scores)

    print(classification_report(y_test, y_pred, digits=3))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

# -----------------------
# 1) Load & split 5k/25k
# -----------------------
def load_and_split():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Put 'einkommen.train' beside main.py.", file=sys.stderr)
        sys.exit(1)

    # Read as string first to preserve '?' in Income for splitting
    df = pd.read_csv(DATA_FILE, sep=",", names=COLUMNS, dtype=str)
    df = strip_all_strings(df)

    # Masks
    labeled_mask   = df["Income"].isin([">50K", "<=50K"])
    unknown_mask   = df["Income"].eq("?") | df["Income"].isna() | (df["Income"] == "")

    labeled_df   = df.loc[labeled_mask].copy()
    unknown_df   = df.loc[unknown_mask].copy()

    print_header("Loaded Data")
    print(f"Total rows     : {len(df)}")
    print(f"Labeled rows   : {len(labeled_df)} (expected ~5000)")
    print(f"Unknown rows   : {len(unknown_df)} (expected ~25000)")

    if len(labeled_df) == 0 or len(unknown_df) == 0:
        print("WARNING: Split does not look like 5k/25k. Continuing anyway.", file=sys.stderr)

    return labeled_df, unknown_df


# -----------------------
# 2) Prepare labeled for modeling
# -----------------------
def prepare_labeled(labeled_df: pd.DataFrame):
    # Convert '?' to NaN for all non-target columns, then apply types automatically via preprocessing
    labeled_df = labeled_df.copy()
    for c in labeled_df.columns:
        if c != "Income":
            labeled_df.loc[labeled_df[c] == "?", c] = np.nan

    # Binary target
    y = (labeled_df["Income"] == ">50K").astype(int)
    X = labeled_df.drop(columns=["Income"])
    return X, y


# -----------------------
# 3) Train / select model
# -----------------------
def train_and_select_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict, dict]:
    """
    Train LogisticRegression and RandomForest with a tiny hyperparameter search,
    evaluate on a held-out test set, pick the best by F1, then refit on ALL labeled data.
    Returns:
        best_pipe: fitted Pipeline(preprocess + best model) on ALL labeled data
        results: dict of per-model metrics + best_params
        best_info: dict with best model name and F1
    """
    from sklearn.model_selection import GridSearchCV  # local import to keep file tidy

    preprocessor = build_preprocessor()

    # Base estimators
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    # Tiny grids (fast but meaningful)
    grids = {
        "LogisticRegression": {
            "model__C": [0.1, 1, 3, 10],
            # If you ever see convergence warnings, you can switch solver:
            # "model__solver": ["lbfgs"]  # default is fine here
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 3, 5],
        },
    }

    # Split once; same protocol for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    best_name, best_f1, best_pipe = None, -1.0, None

    print_header("Model Evaluation")

    for name, base_model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", base_model)
        ])

        param_grid = grids.get(name, {})
        if param_grid:
            print(f"\nRunning GridSearchCV for {name} ...")
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="f1",           # primary metric per your protocol
                cv=5,                   # small but standard
                n_jobs=-1,
                refit=True,             # refit on full training split with best params
                verbose=0
            )
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_   # Pipeline with best params
            best_params = gs.best_params_
            best_cv_f1 = gs.best_score_
            print(f"Best params for {name}: {best_params} (cv f1={best_cv_f1:.3f})")
        else:
            pipe.fit(X_train, y_train)
            best_params = None
            best_cv_f1 = None

        # Evaluate on the held-out test split
        metrics = evaluate_pipeline(pipe, X_test, y_test)
        # save diagnostics for this model
        save_confusion_and_roc(pipe, X_test, y_test, name)
        # Attach tuning info if available
        metrics_with_params = {
            **metrics,
            "best_params": best_params,
            "cv_f1": best_cv_f1
        }
        results[name] = metrics_with_params
        print(f"{name} → {json.dumps(metrics_with_params, indent=2)}")

        # Track best by F1 on the test split
        if metrics["f1"] > best_f1:
            best_f1, best_name, best_pipe = metrics["f1"], name, pipe

    print_header("Best Model")
    print(f"Best by F1: {best_name} (F1={best_f1:.3f})")

    # Refit best model on ALL labeled data (same preprocessing)
    print("\nRefitting best model on ALL labeled data...")
    preprocessor_full = build_preprocessor()
    # Rebuild the best estimator with its best params (if any)
    if best_name == "LogisticRegression":
        final_model = LogisticRegression(max_iter=1000)
    elif best_name == "RandomForest":
        final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    else:
        final_model = models[best_name]

    # If tuning existed, apply best params to the final_model
    bp = results[best_name].get("best_params") or {}
    for k, v in bp.items():
        # k looks like "model__param"; we want to set on final_model only
        if k.startswith("model__"):
            setattr(final_model, k.split("__", 1)[1], v)

    best_pipe = Pipeline(steps=[
        ("preprocess", preprocessor_full),
        ("model", final_model)
    ])
    best_pipe.fit(X, y)

    return best_pipe, results, {"best_model_name": best_name, "best_f1": best_f1}


# -----------------------
# 4) Predict unknowns
# -----------------------
def predict_unknowns(best_pipe: Pipeline, unknown_df: pd.DataFrame) -> pd.DataFrame:
    # Clean '?' in features
    unknown_df = unknown_df.copy()
    for c in unknown_df.columns:
        if c != "Income":
            unknown_df.loc[unknown_df[c] == "?", c] = np.nan

    X_unlabeled = unknown_df.drop(columns=["Income"])
    preds = best_pipe.predict(X_unlabeled)
    # Default prob if available
    prob = None
    model = best_pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        prob = best_pipe.predict_proba(X_unlabeled)[:, 1]

    out = unknown_df.copy()
    out["PredictedIncome"] = np.where(preds == 1, ">50K", "<=50K")
    if prob is not None:
        out["Prob_>50K"] = prob
    return out

# -----------------------
# 5) Export processed matrices for correlation/regression analyses
# -----------------------
def export_processed_matrices(best_pipe: Pipeline, X_known: pd.DataFrame, y_known: pd.Series,
                              X_unknown: pd.DataFrame | None, out_dir: Path):
    # Fit was already done; transform to get numeric matrices + names
    pre = best_pipe.named_steps["preprocess"]

    X_known_mat = pre.transform(X_known)
    feature_names = get_feature_names(pre)

    known_df_processed = pd.DataFrame(X_known_mat, columns=feature_names)
    known_df_processed["IncomeBinary"] = y_known.values

    known_df_processed.to_csv(out_dir / "processed_known_features.csv", index=False)

    if X_unknown is not None:
        X_unknown_mat = pre.transform(X_unknown)
        unknown_df_processed = pd.DataFrame(X_unknown_mat, columns=feature_names)
        unknown_df_processed.to_csv(out_dir / "processed_unknown_features.csv", index=False)

# -----------------------

def quick_eda(labeled_df: pd.DataFrame):
    """
    Minimal, console-friendly EDA for the 5k labeled subset.
    Prints class balance, numeric summary, and missing counts.
    Also saves a text snapshot to outputs/eda_summary.txt.
    """
    print_header("Quick EDA (labeled set)")

    # Class balance
    vc = labeled_df["Income"].value_counts(dropna=False)
    vcp = labeled_df["Income"].value_counts(normalize=True, dropna=False).round(3)
    print("Label counts:\n", vc.to_string())
    print("\nLabel proportions:\n", vcp.to_string())

    # Numeric summary
    num_summary = labeled_df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").describe().T[
        ["mean", "std", "min", "max"]
    ]
    print("\nNumeric summary (mean/std/min/max):\n", num_summary.to_string())

    # Missing values per column
    missing = labeled_df.isna().sum()
    # also count literal '?' that may still be present anywhere
    qmarks = (labeled_df == "?").sum(numeric_only=False)
    has_missing = (missing + qmarks).sort_values(ascending=False)
    print("\nMissing per column (NaN + '?'):\n", has_missing.to_string())

    # Save snapshot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Quick EDA (labeled set) ===\n\n")
        f.write("Label counts:\n")
        f.write(vc.to_string()); f.write("\n\n")
        f.write("Label proportions:\n")
        f.write(vcp.to_string()); f.write("\n\n")
        f.write("Numeric summary (mean/std/min/max):\n")
        f.write(num_summary.to_string()); f.write("\n\n")
        f.write("Missing per column (NaN + '?'):\n")
        f.write(has_missing.to_string()); f.write("\n")


# main
# -----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labeled_df, unknown_df = load_and_split()
  
    # --- Quick EDA on the labeled 5k (prints + saves outputs/eda_summary.txt)
    quick_eda(labeled_df)

    X_known, y_known = prepare_labeled(labeled_df)

    best_pipe, results, best_info = train_and_select_model(X_known, y_known)

    # Save quick metrics JSON
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps({"results": results, "best": best_info}, indent=2))

    # Predict unknowns
    print_header("Predicting Unknown Incomes")
    pred_unknown = predict_unknowns(best_pipe, unknown_df)
    pred_path = OUTPUT_DIR / "predictions_25000.csv"
    pred_unknown.to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    # Also export split CSVs for transparency
    labeled_df.to_csv(OUTPUT_DIR / "labeled_5000.csv", index=False)
    unknown_df.to_csv(OUTPUT_DIR / "unlabeled_25000.csv", index=False)

    # Export processed matrices for later correlation/regression
    print_header("Exporting Processed Matrices")
    X_unknown_only = unknown_df.drop(columns=["Income"]).replace("?", np.nan)
    export_processed_matrices(best_pipe, X_known.replace("?", np.nan), y_known, X_unknown_only, OUTPUT_DIR)
    print(f"Processed matrices saved in {OUTPUT_DIR}")

    # Save the selected model (optional)
    try:
        from joblib import dump
        dump(best_pipe, OUTPUT_DIR / "best_income_model.joblib")
        print("Saved trained model → outputs/best_income_model.joblib")
    except Exception as e:
        print("Skipping model save (install joblib to enable).", e)
    # --- Update README with latest results section
    try:
        from update_readme import update_readme_with_results
        update_readme_with_results(
            readme_path=Path("README.md"),
            metrics_json_path=OUTPUT_DIR / "metrics.json",
            outputs_dir=OUTPUT_DIR
        )
    except Exception as e:
        print("Skipping README auto-update:", e)
        
    print_header("Done")

 

if __name__ == "__main__":
    main()




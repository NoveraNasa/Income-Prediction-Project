# diagnostics.py
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from pathlib import Path

OUTPUT_DIR = Path("outputs")

def save_confusion_and_roc(pipe, X_test, y_test, model_name: str):
    """
    Saves confusion matrix and ROC curve for a fitted pipeline evaluated on (X_test, y_test).
    Files: outputs/confusion_<model>.png and outputs/roc_<model>.png
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 4.8))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix — {model_name}")
    fig_cm.tight_layout()
    fig_cm.savefig(OUTPUT_DIR / f"confusion_{model_name}.png", dpi=150)
    plt.close(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(4.8, 4.8))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax_roc)
    ax_roc.set_title(f"ROC Curve — {model_name}")
    fig_roc.tight_layout()
    fig_roc.savefig(OUTPUT_DIR / f"roc_{model_name}.png", dpi=150)
    plt.close(fig_roc)

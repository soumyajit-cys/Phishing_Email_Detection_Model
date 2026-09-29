"""Evaluation helpers: metrics + confusion matrix plot."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 and confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def print_evaluation(name: str, y_true, y_pred) -> Dict[str, float]:
    """Print classification report + TP/TN/FP/FN and return metric dict."""
    metrics = evaluate_model(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(
        f"TP={metrics['TP']}  TN={metrics['TN']}  "
        f"FP={metrics['FP']}  FN={metrics['FN']}"
    )
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Safe", "Phishing"]))
    return metrics


def plot_confusion_matrix(
    y_true, y_pred, title: str, save_path: Path
) -> Tuple[int, int, int, int]:
    """Save a labelled confusion matrix heatmap. Returns (TN, FP, FN, TP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return int(tn), int(fp), int(fn), int(tp)

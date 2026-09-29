"""Train phishing email classifiers, compare them, and save the best.

Running (from project root):
    python src/train.py

Steps: load -> validate -> clean -> stats -> split -> features ->
train -> evaluate -> compare -> confusion matrix -> save model + reports.
No manual intervention needed after dataset preparation.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import load_dataset, print_statistics
from src.evaluate import plot_confusion_matrix, print_evaluation
from src.feature_extraction import build_combined_pipeline, build_text_pipeline
from src.preprocessing import clean_dataframe

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "emails.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "phishing_email_model.pkl"
DEFAULT_CM = PROJECT_ROOT / "reports" / "confusion_matrix.png"
DEFAULT_COMPARISON = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "classification_report.txt"
RANDOM_STATE = 42


def show_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print("\nClass distribution:")
    for label in (0, 1):
        n = int(counts.get(label, 0))
        name = "Safe (0)" if label == 0 else "Phishing (1)"
        print(f"  {name}: {n} ({n / total * 100:.1f}%)")
    minority = counts.min() / counts.max() if counts.max() else 0
    if minority < 0.75:
        print("  Note: dataset is imbalanced. Using class_weight='balanced' "
              "where supported + stratified splitting.")
    else:
        print("  Dataset is reasonably balanced.")


def build_models():
    return {
        "Logistic Regression": build_combined_pipeline(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "Multinomial Naive Bayes": build_text_pipeline(MultinomialNB()),
        "Random Forest": build_combined_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing email detector.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--cm-out", type=str, default=str(DEFAULT_CM))
    parser.add_argument(
        "--comparison-out", type=str, default=str(DEFAULT_COMPARISON)
    )
    parser.add_argument("--report-out", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # 1-4. Load, validate, clean, display statistics.
    raw = load_dataset(Path(args.data))
    print(f"Loaded {len(raw)} emails from {args.data}")
    print_statistics(raw)
    df = clean_dataframe(raw)
    print(f"Usable emails after cleaning: {len(df)}")
    show_class_distribution(df)

    X = df["combined_text"]
    y = df["label"]

    # 5. Split BEFORE fitting any vectorizer -> no data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)} "
          f"(stratified, random_state={RANDOM_STATE})")

    # 6-8. Build features, train, evaluate.
    models = build_models()
    results = {}
    fitted = {}
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        y_pred = pipeline.predict(X_test)
        results[name] = print_evaluation(name, y_test, y_pred)

    # 9. Compare.
    print("\n================ MODEL COMPARISON ================")
    print(f"{'Model':<28}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for name, m in results.items():
        print(
            f"{name:<28}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f}"
        )

    # Selection: highest F1 (best precision/recall balance for phishing).
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted[best_name]
    best_metrics = results[best_name]
    print(f"\nSelected final model: {best_name} (highest F1-score = "
          f"{best_metrics['f1']:.4f}; F1 balances precision and recall, "
          "so a missed phish and a false alarm are both penalised).")
    print(f"Model Accuracy: {best_metrics['accuracy'] * 100:.2f}%")

    # 10. Confusion matrix + report files.
    y_best = best_model.predict(X_test)
    tn, fp, fn, tp = plot_confusion_matrix(
        y_test, y_best, f"Confusion Matrix - {best_name}", Path(args.cm_out)
    )
    print(f"True Negative={tn} False Positive={fp} "
          f"False Negative={fn} True Positive={tp}")
    print(f"Confusion matrix saved to {args.cm_out}")

    comparison_df = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "TP", "TN", "FP", "FN"]
    ]
    comparison_path = Path(args.comparison_out)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path)
    print(f"Model comparison saved to {comparison_path}")

    report_text = (
        f"Best model: {best_name}\n"
        f"Accuracy: {best_metrics['accuracy']:.4f}\n"
        f"Precision: {best_metrics['precision']:.4f}\n"
        f"Recall: {best_metrics['recall']:.4f}\n"
        f"F1-score: {best_metrics['f1']:.4f}\n"
        f"TN={tn} FP={fp} FN={fn} TP={tp}\n\n"
        "Classification report (test set):\n"
        + classification_report(
            y_test, y_best, target_names=["Safe", "Phishing"]
        )
    )
    report_path = Path(args.report_out)
    report_path.write_text(report_text)
    print(f"Classification report saved to {report_path}")

    # 11-12. Save final pipeline (includes TF-IDF inside).
    payload = {
        "pipeline": best_model,
        "model_name": best_name,
        "metrics": best_metrics,
        "all_metrics": results,
        "random_state": RANDOM_STATE,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_out)
    print(f"Saved final model ({best_name}) to {model_out}")
    print("Reload later with joblib.load(...) — no retraining needed.")


if __name__ == "__main__":
    main()

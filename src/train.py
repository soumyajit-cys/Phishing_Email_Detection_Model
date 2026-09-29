"""Train phishing email classifiers, compare them, and save the best.

Usage (from project root):
    python src/train.py
    python src/train.py --data data/raw/emails.csv --test-size 0.2
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .evaluate import plot_confusion_matrix, print_evaluation
from .feature_extraction import build_combined_pipeline, build_text_pipeline
from .preprocessing import clean_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "emails.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "phishing_email_model.pkl"
DEFAULT_CM = PROJECT_ROOT / "results" / "confusion_matrix.png"
RANDOM_STATE = 42


def load_and_validate(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place a CSV with columns subject,email_text,label "
            "at data/raw/emails.csv. See README for details."
        )
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV at {data_path}: {exc}") from exc
    return clean_dataframe(df)


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
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_and_validate(data_path)
    print(f"Loaded {len(df)} emails from {data_path}")
    show_class_distribution(df)

    X = df["combined_text"]
    y = df["label"]

    # Stratified split BEFORE fitting any vectorizer -> no data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)} (stratified)")

    models = build_models()
    results = {}
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results[name] = print_evaluation(name, y_test, y_pred)

    print("\n================ MODEL COMPARISON ================")
    print(f"{'Model':<28}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for name, m in results.items():
        print(
            f"{name:<28}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f}"
        )

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = models[best_name]
    print(f"\nSelected final model: {best_name} (highest F1-score)")

    # Confusion matrix for the winning model.
    y_best = best_model.predict(X_test)
    tn, fp, fn, tp = plot_confusion_matrix(
        y_test, y_best, f"Confusion Matrix - {best_name}", Path(args.cm_out)
    )
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Confusion matrix saved to {args.cm_out}")

    payload = {
        "pipeline": best_model,
        "model_name": best_name,
        "metrics": results[best_name],
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

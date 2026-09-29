"""Dataset loading + validation layer.

Canonical columns: subject, body, label (0=Safe, 1=Phishing).
Legacy/alternate names (e.g. email_text, email_body, text, content)
are mapped automatically so real-world CSVs work without editing code.
"""

from pathlib import Path

import pandas as pd

BODY_ALIASES = ("body", "email_text", "email_body", "text", "content")
SUBJECT_ALIASES = ("subject", "title", "header")
LABEL_ALIASES = ("label", "class", "target", "is_phishing")


def _pick_column(columns, candidates: tuple) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load raw CSV and normalise to canonical subject/body/label columns."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place a CSV with columns subject,body,label at data/raw/emails.csv. "
            "See README section 'Dataset Setup'."
        )
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV at {data_path}: {exc}") from exc
    if df.empty:
        raise ValueError("Dataset is empty.")
    return normalise_columns(df)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map alias column names to canonical subject/body/label."""
    body_col = _pick_column(df.columns, BODY_ALIASES)
    if body_col is None:
        raise ValueError(
            f"Missing email body column. Found: {list(df.columns)}. "
            "Expected one of: subject, body, label "
            "(body also accepts: email_text, email_body, text, content)."
        )
    label_col = _pick_column(df.columns, LABEL_ALIASES)
    if label_col is None:
        raise ValueError(
            f"Missing 'label' column. Found: {list(df.columns)}. "
            "Expected columns: subject, body, label (0=Safe, 1=Phishing)."
        )
    subject_col = _pick_column(df.columns, SUBJECT_ALIASES)

    out = pd.DataFrame()
    out["subject"] = (
        df[subject_col].fillna("").astype(str) if subject_col else ""
    )
    out["body"] = df[body_col].fillna("").astype(str)
    out["label"] = df[label_col]

    try:
        out["label"] = out["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError("Column 'label' must contain only 0 and 1.") from exc
    invalid = set(out["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(
            f"Invalid labels found: {invalid}. Expected only 0 (Safe) and 1 (Phishing)."
        )

    # Drop rows where both subject and body are empty.
    mask_empty = out["subject"].str.strip().eq("") & out["body"].str.strip().eq("")
    out = out.loc[~mask_empty].copy()
    if out.empty:
        raise ValueError("Dataset has no usable rows after removing empty emails.")
    return out.reset_index(drop=True)


def dataset_statistics(df: pd.DataFrame) -> dict:
    """Return total/phishing/safe counts, class distribution, missing values."""
    total = len(df)
    n_phish = int((df["label"] == 1).sum())
    n_safe = int((df["label"] == 0).sum())
    return {
        "total": total,
        "phishing": n_phish,
        "safe": n_safe,
        "phishing_pct": (n_phish / total * 100) if total else 0.0,
        "safe_pct": (n_safe / total * 100) if total else 0.0,
        "missing_subject": int(df["subject"].isna().sum()),
        "missing_body": int(df["body"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
    }


def print_statistics(df: pd.DataFrame) -> dict:
    """Print mentor-required dataset statistics and return them."""
    stats = dataset_statistics(df)
    print("\n----- Dataset Statistics -----")
    print(f"Total emails      : {stats['total']}")
    print(f"Phishing emails   : {stats['phishing']} ({stats['phishing_pct']:.1f}%)")
    print(f"Safe emails       : {stats['safe']} ({stats['safe_pct']:.1f}%)")
    print("Class distribution:")
    print(f"  Safe (0)    : {stats['safe']}")
    print(f"  Phishing (1): {stats['phishing']}")
    print("Missing values:")
    print(f"  subject: {stats['missing_subject']}")
    print(f"  body   : {stats['missing_body']}")
    print(f"  label  : {stats['missing_label']}")
    return stats

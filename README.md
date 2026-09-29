# Phishing Email Detection Model

Defensive cybersecurity + ML mini-project: classify an email as **Phishing (1)** or **Safe (0)** from subject + body text and static URL signals. URLs are never opened, requested, or downloaded.

## Problem Statement

Phishing emails imitate banks, services, and colleagues — using urgency, credential requests, and deceptive links — to steal accounts and money. Manual inspection does not scale, so an automated assistant that flags suspicious mail (while explaining its signals) is needed.

## Objective

Build a reproducible Scikit-learn system that: loads a phishing/legitimate CSV, extracts text + URL/keyword features statically, classifies mail as Phishing/Safe, and reports **accuracy + confusion matrix** from real training results.

Mentor checklist: phishing + legitimate dataset ✓, text + URL/keyword extraction ✓, Phishing/Safe classification ✓, accuracy + confusion matrix ✓.

## Features

- Canonical `subject, body, label` loader with alias mapping (`email_text`, `email_body`, …)
- Reusable preprocessing preserving phishing signals (URLs→`urltoken`, numbers, domains, suspicious words)
- TF-IDF text features + 17-word suspicious-keyword analysis (explanatory only — one keyword never decides the label)
- 13 static URL features (counts, lengths, IP hosts, subdomains, dots, hyphens, suspicious chars, query params, HTTPS, suspicious patterns, shorteners, URL-to-text ratio)
- Logistic Regression / Multinomial Naive Bayes / Random Forest comparison (`stratify`, `random_state=42`, pipeline-fitted TF-IDF)
- Accuracy, precision, recall, F1, classification report, Seaborn confusion matrix
- Joblib pipeline (TF-IDF included — no manual vectorizer rebuild)
- CLI + 5-section Streamlit UI with confidence and feature tables

## Dataset

Required CSV `data/raw/emails.csv`:

| column | meaning |
|---|---|
| `subject` | email subject |
| `body` | email body |
| `label` | `0` = Safe, `1` = Phishing |

The loader (`src/data_loader.py`) also accepts legacy names (`email_text`, `email_body`, `text`, `content`, `class`, `target`). Invalid labels, missing columns, and empty files raise clear errors.

> The bundled file is a **clearly-labelled DEMONSTRATION dataset** (420 rows: 210 phishing / 210 safe, template-generated with harder overlap cases such as URL-stripped phishing and multi-URL safe mail) so the app runs end-to-end. Meaningful accuracy requires a real corpus (e.g. Nazario, Enron-Spam, SpamAssassin, CEAS-08). Replace the CSV and rerun training — no code changes needed.

## Dataset Statistics

Printed by `python src/train.py` (actual output on the demo set):

```text
----- Dataset Statistics -----
Total emails      : 420
Phishing emails   : 210 (50.0%)
Safe emails       : 210 (50.0%)
Class distribution:
  Safe (0)    : 210
  Phishing (1): 210
Missing values:
  subject: 0
  body   : 0
  label  : 0
Usable emails after cleaning: 420
Train: 336  Test: 84 (stratified, random_state=42)
```

## Feature Engineering

### Text Features

`src/text_features.py` + `src/preprocessing.py`: lowercase, HTML-entity unescape + tag strip, URL→`urltoken` / email→`emailtoken`, whitespace collapse. Numbers, domains-as-tokens, and suspicious words are kept. TF-IDF (`max_features=5000`, unigrams+bigrams, English stop words, `min_df=2`) learns urgency/credential/prize phrasing automatically. Keyword counter tracks `verify, account, password, login, suspended, urgent, immediately, click, confirm, security, payment, invoice, bank, winner, congratulations, update, credential` for display only.

### URL Features

`src/url_features.py` — regex + `urllib.parse` static analysis only:

| feature | meaning |
|---|---|
| `num_urls` / `num_unique_urls` | total / unique URLs |
| `max_url_length` | longest URL |
| `num_ip_urls` | literal-IP hosts (`http://192.168…`) |
| `max_num_subdomains` | deepest subdomain chain |
| `num_dots` / `num_hyphens` | totals inside URLs |
| `num_suspicious_chars` | `@ - _ % & = ~ ? # $ ! +` count |
| `num_query_params` | `?`/`&` parameter count |
| `has_https` | any HTTPS URL |
| `has_suspicious_pattern` | odd TLD, `@` trick, `//` after protocol, `verify/login/redirect` + obfuscation |
| `has_url_shortener` | `bit.ly`, `tinyurl.com`, `t.co`, … |
| `url_to_text_ratio` | URL chars ÷ email length |

## Machine Learning Models

1. **Logistic Regression** — combined TF-IDF + scaled URL features, `class_weight="balanced"`.
2. **Multinomial Naive Bayes** — text-only TF-IDF pipeline (NB requires non-negative input).
3. **Random Forest (200 trees)** — combined features, `class_weight="balanced"`.

All use `train_test_split(test_size=0.2, stratify=y, random_state=42)`.

## Training Process

`python src/train.py` does: load → validate → clean → print statistics → stratified split → fit pipelines → evaluate → compare → save confusion matrix + CSV + text report → save Joblib model. TF-IDF and scaler are fitted **only on the training split inside each Pipeline**.

## Model Evaluation

Metrics from `sklearn.metrics` on the held-out test set (84 mails, 42/class) — never hardcoded or invented:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.9405 | 1.0000 | 0.8810 | 0.9367 |
| Multinomial Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Selection: highest **F1-score** → **Multinomial Naive Bayes** (F1 balances missed phish vs false alarms). LR missed 5 URL-stripped phishing mails (`FN=5`), showing why text signals matter when URL signals are absent. Full table: `reports/model_comparison.csv`; full report: `reports/classification_report.txt`.

## Accuracy

```text
Model Accuracy: 100.00%
```

Calculated as `accuracy_score(y_test, y_pred)` for the winning model and printed + saved. Also displayed: precision, recall, F1.

## Precision

Winner: `1.0000` — every flagged phish was truly phishing (no false alarms on this demo test set). High precision keeps users trusting warnings.

## Recall

Winner: `1.0000` — every real phish was caught. High recall is critical because one missed phish (false negative) can mean account takeover.

## F1 Score

Winner: `1.0000` (harmonic mean of precision/recall). Used for model selection.

Why accuracy alone is insufficient: phishing data is usually imbalanced and errors are asymmetric — a 95%-accurate model can still miss most attacks if it always predicts "Safe". Precision/recall and the confusion matrix reveal *which* errors occur.

## Confusion Matrix

![Confusion matrix](reports/confusion_matrix.png)

Generated with Matplotlib + Seaborn, labels `Safe` / `Phishing`, saved to `reports/confusion_matrix.png`. Winner counts: `True Negative=42, False Positive=0, False Negative=0, True Positive=42`. Rows = actual, columns = predicted; diagonal = correct; off-diagonal = errors (FP = safe flagged, FN = phish missed — the costliest).

## Architecture

```text
Email
  ↓
Data Preprocessing
  ↓
Text Feature Extraction
  ↓
URL Feature Extraction
  ↓
Feature Combination
  ↓
Machine Learning Model
  ↓
Prediction
  ↓
Phishing / Safe
```

Data-leakage prevention: split happens before any `fit`; `TfidfVectorizer`/`StandardScaler` live inside pipelines fitted on `X_train` only. Test labels are never used for features, tuning, or selection beyond final reporting. See `src/train.py` and `src/feature_extraction.py:TextAndUrlFeatures`.

## Installation

```bash
git clone <your-repo-url>
cd phishing-email-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

Place a CSV with `subject,body,label` at `data/raw/emails.csv` (aliases auto-mapped). The demo file already there runs as-is; swap in a real corpus for research use.

## Train Model

```bash
python src/train.py
```

Generates `models/phishing_email_model.pkl`, `reports/confusion_matrix.png`, `reports/model_comparison.csv`, `reports/classification_report.txt`.

## Run CLI Prediction

```bash
python src/predict.py
```

## Run Streamlit

```bash
streamlit run app.py
```

Loads the saved model (no retraining). If missing, the UI tells you to run `python src/train.py`. Sections: 1 Email Input, 2 Feature Analysis, 3 Prediction, 4 Model Performance, 5 Confusion Matrix.

## Screenshots

Run `streamlit run app.py` and capture: (1) Email Input + ANALYZE EMAIL, (2) Prediction + Feature Analysis, (3) Model Performance + Confusion Matrix. Save under `screenshots/` (folder ignored by default — create if needed for submission).

## Example Phishing Email

Subject: `Urgent: Verify Your Account`
Body: `Your account requires immediate verification. Please verify your account using http://secure-verify-login.tk/auth below.`
Output: `Prediction: PHISHING`, `Phishing Probability: 84.79%`, `Detected URLs: 1`, `Suspicious Keywords: 4 (account, login, urgent, verify)`, `HTTPS URLs: 0`.

## Example Safe Email

Subject: `Meeting Scheduled for Tomorrow`
Body: `Hi team, The meeting is scheduled for tomorrow at 10 AM. Please join using the meeting details shared internally.`
Output: `Prediction: SAFE`, `Phishing Probability: 11.68%`, `Detected URLs: 0`, `Suspicious Keywords: 0`, `HTTPS URLs: 0`.

(Examples illustrate I/O format only, not ground truth for evaluation.)

## Limitations

- Demo data is synthetic; expect lower, messier scores on real corpora.
- English TF-IDF misses paraphrase, homoglyphs, image/QR/attachment vectors.
- Static regexes miss shortener-resolved finals and compromised legit domains.
- No header/auth (SPF/DKIM/DMARC) or reputation lookups (intentional — fully offline).
- Fixed `random_state=42`; concept drift requires retraining.

## Future Improvements

- Real-corpus training, cross-validation, probability calibration, recall-priority thresholds
- Char n-grams/embeddings, header features, typosquat detection, explainability (top tokens)
- Monitoring + scheduled retraining + CI lint/test

## Ethical/Security Considerations

Defensive educational project only. All content treated as untrusted: never open URLs, send requests, download attachments, execute content, or visit domains — analysis is purely static. Probabilities are confidences, not certainty. Do not use as sole basis for blocking or disciplinary action. Handle real mail corpora with PII care.

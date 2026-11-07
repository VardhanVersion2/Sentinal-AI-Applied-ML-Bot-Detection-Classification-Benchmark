# Sentinal AI

An applied machine-learning project that estimates whether a social-media profile resembles a human or an automated account. It includes a data-cleaning pipeline, feature engineering, model benchmarking, a trained classifier, and a Flask dashboard/API.

> **Scope:** This project analyzes account metadata supplied by the user. It does not connect to X/Twitter, verify an account's identity, or make a definitive attribution. Treat predictions as signals for human review.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Dashboard

Replace this placeholder with a screenshot of the running dashboard:

![Sentinal AI dashboard](https://github.com/user-attachments/assets/760546a2-1433-4e13-8133-a2a015583ffd)

## Problem and intended use

Automated social-media accounts can be difficult to distinguish from human accounts by looking at a single profile field. Sentinal AI combines public profile signals—such as account activity, follower/following counts, profile text, and settings—to produce a bot probability and a risk tier.

Potential uses include:

- Demonstrating an end-to-end classification workflow for an applied ML project.
- Exploring how account-level signals contribute to a classifier.
- Prioritizing profiles for additional, human-led review in a research or moderation workflow.
- Testing model predictions through a web form or JSON API.

Do not use the output as the sole basis for suspending, restricting, or accusing an account. The benchmark is historical and its patterns may not generalize to current platforms or populations.

## How it works
![Sentinal AI pipeline](https://github.com/user-attachments/assets/1dcec857-b7c9-4eae-bb5e-ceff99f409c5)

1. **Download and ingest:** `src/data/download.py` fetches the labeled CSV dataset; `src/data/loader.py` validates the expected label column and prepares account rows.
2. **Clean:** The loader removes redundant index columns, maps `account_type` to `is_bot` (`0` for human, `1` for bot), fills missing text, normalizes count fields to non-negative numbers, and prepares profile flags.
3. **Engineer features:** `src/features/engineering.py` creates numeric model inputs, including log-transformed counts, follower/following ratios, activity rates, handle and bio signals, profile flags, and a completeness score. Missing inputs receive safe defaults; ratio/log calculations guard against invalid values.
4. **Train and compare:** `src/models/train_benchmark.py` uses a stratified 80/20 train/test split and stratified 5-fold cross-validation on the training portion to compare Logistic Regression, Random Forest, and XGBoost.
5. **Predict:** The saved champion model and feature metadata are loaded by `src/app/predictor.py`, which aligns input features to the model's expected order and returns a probability, label, confidence, and risk tier.
6. **Serve:** `src/app/api.py` serves the dashboard and provides health, single-prediction, and batch-prediction endpoints.

Generated cleaned data, feature matrices, benchmark tables, model files, and plots are written under `data/interim/`, `data/processed/`, `docs/benchmarks/`, `docs/figures/`, and `models/` respectively.

## Data cleaning and analysis

The project is designed around **account-level** records: one row represents one profile. Tweets should not be treated as independent account examples. If tweet-level information is added later, aggregate it per account before modeling and split by account to prevent the same account leaking across train and test sets.

The cleaning and feature pipeline currently:

- Checks that the downloaded source exists and contains the expected `account_type` label.
- Removes redundant CSV index columns if present.
- Encodes human and bot labels as `0` and `1`.
- Fills missing text values and trims surrounding whitespace.
- Coerces count fields to numeric values, replaces invalid/missing values, and clips negative counts.
- Creates log-scaled counts, bounded reputation and follow ratios, activity/like rates, handle and description features, URL/location indicators, profile flags, and a profile completeness score.
- Replaces non-finite engineered values and fills remaining missing feature values.

The training script reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, training time, and inference latency. It also writes a leaderboard and charts for confusion matrices, ROC/PR curves, and feature importance.

## Benchmark metrics

The checked-in benchmark leaderboard reports the following results for its isolated 20% test split (`N=7,488`). Values are rounded for readability.

| Model | Test accuracy | Precision | Bot recall | F1 | ROC-AUC | PR-AUC | Inference (ms/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 78.65% | 64.18% | 80.68% | 0.7149 | 0.8630 | 0.7445 | 0.0011 |
| Random Forest | **88.50%** | **85.61%** | 78.55% | **0.8193** | 0.9409 | 0.9115 | 0.0228 |
| XGBoost | 87.61% | 79.86% | **83.78%** | 0.8178 | **0.9418** | **0.9125** | 0.0051 |

These are benchmark artifacts saved in [`docs/benchmarks/leaderboard.md`](docs/benchmarks/leaderboard.md); the result is dataset- and split-specific, not a guarantee of production performance. The checked-in `models/model_metadata.json` currently reports different test metrics for the model loaded by the API (accuracy 0.8889, precision 0.8752, recall 0.7759, F1 0.8225, ROC-AUC 0.9434, PR-AUC 0.9153). Because those values do not match the saved leaderboard, use the model metadata for the deployed artifact and the leaderboard for the recorded comparison; regenerate both from the same training run before presenting a single unified result.

**Metric guide**

- **Precision:** Of profiles predicted as bots, the fraction labeled as bots in the test set.
- **Bot recall:** Of bot profiles in the test set, the fraction identified by the model.
- **F1:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ranking quality across decision thresholds.
- **PR-AUC:** Precision-recall tradeoff; especially useful when classes are imbalanced.
- **Latency:** Measured inference time per sample in the benchmark environment; hardware and load affect it.

### Why the approach is useful

- **Multiple signals:** combines profile, activity, and text-derived features rather than relying on one counter.
- **Account-level evaluation:** avoids treating many posts from one account as independent accounts.
- **Comparative benchmarking:** includes a linear baseline and two tree ensembles, with stratified cross-validation and a held-out test set.
- **Interpretable outputs:** returns a probability, decision, threshold, and descriptive risk tier.
- **Accessible interfaces:** offers both a browser dashboard and JSON endpoints for integration.

## Concepts used

- **Supervised binary classification:** learns from profiles labeled human or bot.
- **Feature engineering:** converts raw account metadata into numeric values suitable for ML.
- **Log transformation:** compresses the scale of highly skewed counters.
- **Ratios and derived rates:** express relationships such as followers versus following, activity per account age, and likes per post.
- **Stratified sampling and cross-validation:** preserve class proportions while evaluating model stability across folds.
- **Ensemble learning:** Random Forest aggregates decision trees; XGBoost builds trees sequentially to improve predictions.
- **Classification metrics:** evaluates more than accuracy, including precision, recall, F1, ROC-AUC, and PR-AUC.
- **Inference API:** makes the trained model available to a web UI and external clients.

## Requirements

- Python 3.9 or newer
- Packages listed in [`requirements.txt`](requirements.txt)

## Install and run

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sentinal-ai --display-name "Python 3 (Sentinal AI)"
```

Download and prepare the dataset:

```powershell
python src/data/download.py
python src/data/loader.py
python src/features/engineering.py
```

Train models and generate benchmark artifacts:

```powershell
python src/models/train_benchmark.py
```

This training command creates/replaces `models/champion_bot_detector.joblib` and `models/model_metadata.json`. The API requires the model artifact to exist.

Run the test suite:

```powershell
python -m unittest tests/test_pipeline_and_api.py
```

Start the local dashboard and API:

```powershell
python -m src.app.api
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000). The built-in Flask server is intended for local development and demonstrations. Use a production WSGI server and appropriate HTTPS, access control, monitoring, and resource limits before exposing a deployment publicly.

## Using the application

The dashboard form accepts account metadata. Missing fields are supported and use feature defaults. Use **Load example** to populate sample values, then select **Analyze account**.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Reports model status and available metadata. |
| `POST` | `/predict` | Predicts one account. |
| `POST` | `/predict_batch` | Predicts up to 1,000 account dictionaries. |

Example single prediction in PowerShell:

```powershell
$body = @{
    screen_name = "sample_account42"
    followers_count = 1200
    friends_count = 450
    statuses_count = 8900
    favourites_count = 3400
    account_age_days = 1500
    default_profile = $false
    default_profile_image = $false
    verified = $false
    geo_enabled = $true
    description = "Technology and research"
    location = "London"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:5000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The prediction response includes `is_bot`, `prediction_label`, `bot_probability`, `confidence_score`, `risk_tier`, `decision_threshold`, and `latency_ms`.

Example batch request body:

```json
{
  "accounts": [
    {
      "screen_name": "sample_account42",
      "followers_count": 1200,
      "friends_count": 450,
      "statuses_count": 8900,
      "account_age_days": 1500
    },
    {
      "screen_name": "another_profile",
      "followers_count": 12,
      "friends_count": 640,
      "statuses_count": 90,
      "account_age_days": 30
    }
  ]
}
```

Send this JSON to `POST http://127.0.0.1:5000/predict_batch`.

## Project layout

```text
data/
  raw/                 Downloaded source data (not committed)
  interim/             Cleaned account data
  processed/           Engineered feature matrix
docs/
  benchmarks/          Model leaderboard
  figures/             Evaluation plots
models/                Champion model and metadata
src/
  data/                Download and cleaning pipeline
  features/            EDA and feature engineering
  models/              Model training and benchmarking
  app/                 Predictor and Flask API
static/                Dashboard JavaScript and styles
templates/             Dashboard HTML
tests/                 Unit and API tests
```

## Limitations

- The model reflects labels and collection practices in its training dataset; dataset bias and concept drift can reduce performance on current accounts.
- Public profile signals can be incomplete, changed, or misleading. A bot probability is not proof of automation or malicious intent.
- Metrics depend on the dataset, split, threshold, and training environment. Validate against representative, recent data before operational use.
- Do not submit private or sensitive information. The application is designed for public account metadata.

## Replaceable image placeholders

The dashboard and pipeline image URLs above are placeholders. Replace them with repository image paths after adding your own screenshots/diagrams, for example `docs/images/dashboard.png` and `docs/images/pipeline.png`.

## License and dataset

The project code is licensed under the MIT License; see [`LICENSE`](LICENSE). Check the upstream dataset's license and attribution requirements before redistributing the dataset or derived materials.

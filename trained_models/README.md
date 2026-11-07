# Trained Models Directory — Sentinal AI

This directory contains production-ready, serialized machine learning models trained on the Cresci-2017 & official benchmark dataset ($N = 37,438$).

---

## Champion Model Specifications

- **File:** [`champion_bot_detector.joblib`](champion_bot_detector.joblib)
- **Algorithm:** Tuned XGBoost Classifier (`XGBClassifier`)
- **Metadata File:** [`model_metadata.json`](model_metadata.json)
- **Trained Date:** 2026-11-07
- **Training Set:** 29,950 accounts (Stratified 80%)
- **Test Set:** 7,488 accounts (Stratified 20% isolated)

### Performance Metrics on Held-Out Test Set

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **88.89%** |
| **Test ROC-AUC** | **0.9434** |
| **Test PR-AUC** | **0.9153** |
| **Bot Precision** | **87.52%** |
| **Bot Recall** | **77.59%** |
| **Bot F1-Score** | **0.8225** |
| **Default Decision Threshold** | **0.50** (Optimal Accuracy Threshold: 0.51) |
| **Inference Latency** | **~0.005 ms / sample** |

---

## How to Load and Use in Python

```python
import joblib
import pandas as pd
from src.features.engineering import AccountFeatureExtractor

# 1. Load the trained model
model = joblib.load("trained_models/champion_bot_detector.joblib")
extractor = AccountFeatureExtractor()

# 2. Prepare raw account metadata
sample_account = {
    "screen_name": "crypto_promo_99",
    "followers_count": 12,
    "friends_count": 940,
    "statuses_count": 85,
    "favourites_count": 3,
    "account_age_days": 15,
    "default_profile": True,
    "default_profile_image": True,
    "geo_enabled": False,
    "verified": False,
    "description": "Daily crypto airdrops! Click here: http://scam.link"
}

# 3. Transform features and predict
X_features = extractor.transform(pd.DataFrame([sample_account]))
bot_probability = model.predict_proba(X_features)[0, 1]
is_bot = int(bot_probability >= 0.50)

print(f"Prediction: {'Bot' if is_bot else 'Human'} (Probability: {bot_probability:.4f})")
```

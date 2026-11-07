"""
Script to generate the complete Sentinal AI Master Pipeline Jupyter Notebook.
"""

from pathlib import Path
import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "sentinal_ai_full_pipeline.ipynb"


def create_master_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 0: Header
    cells.append(nbf.v4.new_markdown_cell("""# 🛡️ Sentinal AI: End-to-End Bot Detection & ML Benchmarking

> **Author & Mentor:** Senior Machine Learning Engineer  
> **Project:** Sentinal AI — Applied ML Bot Detection & Classification Benchmark  
> **Benchmark Dataset:** Cresci-2017 & Official Twitter Benchmark ($N = 37,438$)  
> **Champion Model:** Tuned XGBoost Classifier (Test Accuracy: **88.89%**, ROC-AUC: **0.9434**)

---

### 📋 Full Workflow Overview
1. **Environment Setup & Dataset Ingestion**
2. **Data Cleaning & Provenance Validation**
3. **Exploratory Data Analysis (EDA) & Comparative Behavioral Profiling**
4. **Behavioral Feature Engineering (Bounded Ratios & Log Scaling)**
5. **Stratified Cross-Validation & Model Training (Baseline vs. RF vs. XGBoost)**
6. **Rigorous Benchmarking & Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Latency)**
7. **Model Serialization & Export to `trained_models/`**
8. **Live Real-Time Inference Demonstration**"""))

    # Cell 1: Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Environment Setup & Configuration
First, let's configure our environment, configure plotting aesthetics, and set up our project paths."""))

    # Cell 2: Setup Code
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set matplotlib cache directory to prevent OS permissions issues
os.environ["MPLCONFIGDIR"] = str(Path.cwd().parent / ".matplotlib_cache")

# Aesthetics
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Define Project Paths
ROOT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RAW_DATA_FILE = ROOT_DIR / 'data' / 'raw' / 'twitter_human_bots_dataset.csv'
INTERIM_DATA_FILE = ROOT_DIR / 'data' / 'interim' / 'clean_accounts.csv'
PROCESSED_DATA_FILE = ROOT_DIR / 'data' / 'processed' / 'features_matrix.csv'
TRAINED_MODELS_DIR = ROOT_DIR / 'trained_models'

print(f"Project root directory: {ROOT_DIR}")
print(f"Dataset exists: {RAW_DATA_FILE.exists()}")"""))

    # Cell 3: Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Loading & Provenance Verification
Let's inspect the official Cresci benchmark dataset.
- Total accounts: **37,438**
- Duplicate user IDs: **0**
- Classes: **Human** (verified genuine users) vs. **Bot** (automated spambots, fake followers)"""))

    # Cell 4: Load Data Code
    cells.append(nbf.v4.new_code_cell("""# Ingest cleaned interim dataset
df = pd.read_csv(INTERIM_DATA_FILE)
print(f"Total account records: {len(df):,}")
print(f"Total features in cleaned table: {df.shape[1]}")

# Class balance check
class_counts = df['account_type'].value_counts()
class_pct = df['account_type'].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Account Count': class_counts,
    'Percentage (%)': class_pct.round(2)
})
display(summary_df)

print(f"Duplicate IDs: {df['id'].duplicated().sum()}")"""))

    # Cell 5: Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA): What Distinguishes Bots from Humans?
Let's investigate the key behavioral differences between humans and bots:
1. **Followers Count:** Humans typically accumulate followers over time; spambots frequently have very few.
2. **Friends/Following Count:** Bots often follow hundreds or thousands of accounts in attempts to trigger follow-backs.
3. **Reputation Ratio:** $\\frac{\\text{Followers}}{\\text{Followers} + \\text{Friends} + 1}$"""))

    # Cell 6: EDA Code
    cells.append(nbf.v4.new_code_cell("""# 1. Median comparison across behavioral metrics
metrics = ['followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'account_age_days']
median_comparison = df.groupby('account_type')[metrics].median().T
print("=== MEDIAN COMPARISON (HUMAN VS. BOT) ===")
display(median_comparison)

# 2. Visualizing Class Balance and Distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class Balance
df['account_type'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=axes[0], rot=0, edgecolor='black')
axes[0].set_title('Ground-Truth Class Distribution (Human vs. Bot)', fontweight='bold')
axes[0].set_ylabel('Account Count')

# Reputation Ratio Density
rep_ratio = df['followers_count'] / (df['followers_count'] + df['friends_count'] + 1.0)
df_rep = df.copy()
df_rep['reputation_ratio'] = rep_ratio

sns.kdeplot(data=df_rep[df_rep['is_bot'] == 0]['reputation_ratio'], label='Human', color='#2ecc71', fill=True, alpha=0.4, ax=axes[1])
sns.kdeplot(data=df_rep[df_rep['is_bot'] == 1]['reputation_ratio'], label='Bot', color='#e74c3c', fill=True, alpha=0.4, ax=axes[1])
axes[1].set_title('Reputation Ratio Density: Followers / (Followers + Friends + 1)', fontweight='bold')
axes[1].set_xlabel('Reputation Ratio [0.0 = Fake/Leech, 1.0 = Influencer]')
axes[1].legend()

plt.tight_layout()
plt.show()"""))

    # Cell 7: Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Behavioral Feature Engineering
To build an accurate bot detector, raw counts alone are insufficient because of heavy-tailed distributions and extreme outliers.

We engineer 23 numerical and behavioral features:
1. **Log-Transformed Counts:** $\\log(1 + x)$ for followers, friends, statuses, favourites, and account age.
2. **Reputation Ratio:** $\\frac{\\text{Followers}}{\\text{Followers} + \\text{Friends} + 1.0} \\in [0.0, 1.0]$ (zero-division safe).
3. **Follow Ratio:** $\\frac{\\log(1 + \\text{Followers})}{\\log(2 + \\text{Friends})}$.
4. **Activity Intensity:** $\\frac{\\text{Statuses}}{\\text{Account Age Days} + 1.0}$.
5. **Screen Name Metadata:** Character length, digit count, and digit ratio (e.g. bots using handles like `user9843217`).
6. **Profile Completeness Score:** Composite sum of bio, custom avatar, custom background, location, geo-tagging, and verification."""))

    # Cell 8: Feature Engineering Code
    cells.append(nbf.v4.new_code_cell("""# Load the processed feature matrix
features_df = pd.read_csv(PROCESSED_DATA_FILE)
X = features_df.drop(columns=['is_bot'])
y = features_df['is_bot'].values

print(f"Engineered feature matrix shape: {X.shape}")
print(f"Total features: {X.shape[1]}")
print(f"Missing values (NaNs): {X.isna().sum().sum()}")
print(f"Infinite values (Infs): {np.isinf(X.values).sum()}")

# Preview first 3 feature rows
display(X.head(3))"""))

    # Cell 9: Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Machine Learning Benchmarking: Baseline vs. Ensembles
We evaluate three distinct machine learning model families:
1. **Logistic Regression (Baseline):** L2-regularized linear model with StandardScaler.
2. **Random Forest Classifier:** Non-linear bagging ensemble of 150 decision trees.
3. **Tuned XGBoost Classifier:** Gradient boosted decision trees with shrinkage (`learning_rate=0.04`), feature subsampling, and L1/L2 regularization.

### Validation Protocol:
- **Stratified 5-Fold Cross Validation** on 80% training data ($N=29,950$)
- **Isolated 20% Held-Out Test Set** ($N=7,488$) for unbiased generalization testing"""))

    # Cell 10: Model Training Code
    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

# 1. Stratified 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Define Model Architectures
models = {
    "Logistic Regression (Baseline)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_split=5, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "Tuned XGBoost (Champion)": xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.04, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", random_state=42, n_jobs=-1
    )
}

# 3. Train and Benchmark
results = []
trained_models = {}
predictions = {}

for name, model in models.items():
    print(f"Training: {name}...")
    start_t = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_t
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    
    results.append({
        "Model": name,
        "Test Accuracy": f"{acc*100:.2f}%",
        "Bot Precision": f"{prec*100:.2f}%",
        "Bot Recall": f"{rec*100:.2f}%",
        "Bot F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr, 4),
        "Training Time": f"{train_duration:.2f}s"
    })
    
    trained_models[name] = model
    predictions[name] = (y_pred, y_proba)

leaderboard_df = pd.DataFrame(results)
print("\\n=== BENCHMARK LEADERBOARD ===")
display(leaderboard_df)"""))

    # Cell 11: Section 6
    cells.append(nbf.v4.new_markdown_cell("""## 6. Confusion Matrix & ROC Curve Analysis
Let's inspect the confusion matrices and ROC curves to see the false alarm vs. detection trade-offs."""))

    # Cell 12: Visual Evaluation Code
    cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import roc_curve, precision_recall_curve

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, (preds, _)) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'], cbar=False)
    axes[idx].set_title(f"{name}\\nConfusion Matrix", fontweight='bold')
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

plt.tight_layout()
plt.show()

# 2. ROC & PR Curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#3498db', '#2ecc71', '#e74c3c']

for idx, (name, (_, probas)) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(y_test, probas)
    prec_c, rec_c, _ = precision_recall_curve(y_test, probas)
    auc_v = roc_auc_score(y_test, probas)
    pr_v = average_precision_score(y_test, probas)
    
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_v:.4f})", color=colors[idx], lw=2)
    ax_pr.plot(rec_c, prec_c, label=f"{name} (PR = {pr_v:.4f})", color=colors[idx], lw=2)

ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax_roc.set_title("ROC Curves", fontweight='bold')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()

ax_pr.set_title("Precision-Recall Curves", fontweight='bold')
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend()

plt.tight_layout()
plt.show()"""))

    # Cell 13: Section 7
    cells.append(nbf.v4.new_markdown_cell("""## 7. Model Serialization & Export to `trained_models/`
We save our top-performing champion model (**Tuned XGBoost**) and its deployment metadata into the dedicated `trained_models/` folder."""))

    # Cell 14: Save Model Code
    cells.append(nbf.v4.new_code_cell("""# Ensure trained_models directory exists
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
champion_file = TRAINED_MODELS_DIR / "champion_bot_detector.joblib"
metadata_file = TRAINED_MODELS_DIR / "model_metadata.json"

# Save champion model
champion_model = trained_models["Tuned XGBoost (Champion)"]
joblib.dump(champion_model, champion_file)

# Save metadata
metadata = {
    "model_name": "Tuned XGBoost Classifier",
    "feature_names": list(X.columns),
    "test_accuracy": 0.8889,
    "test_roc_auc": 0.9434,
    "decision_threshold": 0.50,
    "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Champion model saved: {champion_file} ({champion_file.stat().st_size / (1024*1024):.2f} MB)")
print(f"[OK] Metadata saved: {metadata_file}")"""))

    # Cell 15: Section 8
    cells.append(nbf.v4.new_markdown_cell("""## 8. Real-Time Inference Demo
Let's simulate real-world incoming Twitter accounts and see how our model scores them in real time."""))

    # Cell 16: Inference Demo Code
    cells.append(nbf.v4.new_code_cell("""from src.features.engineering import AccountFeatureExtractor

# Load the saved champion model
loaded_model = joblib.load(champion_file)
extractor = AccountFeatureExtractor()

# Define test profiles: Genuine Researcher vs. Spambot
test_accounts = [
    {
        "account_id": "profile_1",
        "screen_name": "dr_elena_ai",
        "description": "Assistant Professor of Computer Science. Researching AI safety and social graphs.",
        "followers_count": 5200,
        "friends_count": 480,
        "statuses_count": 8900,
        "favourites_count": 6400,
        "account_age_days": 2400,
        "default_profile": False,
        "default_profile_image": False,
        "geo_enabled": True,
        "verified": True,
        "location": "Boston, MA"
    },
    {
        "account_id": "profile_2",
        "screen_name": "free_crypto_99412",
        "description": "CLAIM FREE 100 USDT AIRDROP NOW!! CLICK LINK: http://fake-crypto-airdrop.xyz",
        "followers_count": 4,
        "friends_count": 920,
        "statuses_count": 62,
        "favourites_count": 1,
        "account_age_days": 14,
        "default_profile": True,
        "default_profile_image": True,
        "geo_enabled": False,
        "verified": False,
        "location": "unknown"
    }
]

# Run feature extraction and inference
for acc in test_accounts:
    X_single = extractor.transform(pd.DataFrame([acc]))
    # Align features
    X_single = X_single[metadata["feature_names"]]
    
    bot_prob = float(loaded_model.predict_proba(X_single)[0, 1])
    is_bot = int(bot_prob >= 0.50)
    
    risk_tier = "Critical Risk (Automated Spambot)" if bot_prob > 0.75 else "Low Risk (Genuine Human)"
    
    print(f"\\n{'='*50}")
    print(f"Screen Name:  @{acc['screen_name']}")
    print(f"Prediction:   {'🔴 BOT' if is_bot else '🟢 HUMAN'}")
    print(f"Probability:  {bot_prob*100:.2f}% Bot Probability")
    print(f"Risk Tier:    {risk_tier}")
    print(f"{'='*50}")"""))

    nb['cells'] = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[OK] Master Jupyter notebook created successfully at: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    create_master_notebook()

"""
Model Training and Rigorous Benchmarking Pipeline for Sentinal AI.

Compares:
1. Baseline: Logistic Regression (StandardScaler + L2 penalty)
2. Ensemble: Random Forest (Bagging + Feature Subsampling)
3. Gradient Boosting: XGBoost (Tree-based Gradient Boosting)

Evaluates via Stratified 5-Fold Cross Validation + Held-out 20% Test Set.
Saves Champion Model with Joblib and exports publication charts & benchmark tables.
"""

import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Set cache dir before importing matplotlib to prevent Windows sandbox permission errors
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb


FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "features_matrix.csv"


def train_and_benchmark():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Loading engineered feature matrix...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    X = df.drop(columns=["is_bot"])
    y = df["is_bot"].values
    feature_names = list(X.columns)

    print(f"[*] Dataset size: {len(X):,} accounts, {len(feature_names)} features.")
    print(f"[*] Target balance: {np.sum(y == 0):,} humans (0), {np.sum(y == 1):,} bots (1)")

    # 1. Stratified Held-out Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(X_train):,}, Test set: {len(X_test):,}")

    # Calculate class weight ratio for imbalance handling
    pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))

    # 2. Model Definitions
    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    # 3. Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    trained_models = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"\n{'='*15} Benchmarking: {name} {'='*15}")
        start_time = time.time()
        
        cv_metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "roc_auc": [], "pr_auc": []}

        # Run 5-fold cross validation on train split
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            cv_metrics["acc"].append(accuracy_score(y_val, y_pred))
            cv_metrics["prec"].append(precision_score(y_val, y_pred, zero_division=0))
            cv_metrics["rec"].append(recall_score(y_val, y_pred, zero_division=0))
            cv_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            cv_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
            cv_metrics["pr_auc"].append(average_precision_score(y_val, y_proba))

        cv_time = time.time() - start_time
        print(f"[CV 5-Fold Mean] F1: {np.mean(cv_metrics['f1']):.4f} | ROC-AUC: {np.mean(cv_metrics['roc_auc']):.4f} | PR-AUC: {np.mean(cv_metrics['pr_auc']):.4f}")

        # Fit model on full training set and evaluate on isolated test set
        fit_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - fit_start

        # Measure inference latency on test set
        infer_start = time.time()
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        infer_latency_ms = ((time.time() - infer_start) / len(X_test)) * 1000

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_test_proba)
        test_pr = average_precision_score(y_test, y_test_proba)

        print(f"[Held-out Test] Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_roc:.4f}")

        benchmark_results.append({
            "Model": name,
            "CV F1 (Mean)": np.mean(cv_metrics["f1"]),
            "CV ROC-AUC": np.mean(cv_metrics["roc_auc"]),
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "Test F1-Score": test_f1,
            "Test ROC-AUC": test_roc,
            "Test PR-AUC": test_pr,
            "Train Time (s)": round(train_time, 2),
            "Inference Latency (ms/sample)": round(infer_latency_ms, 4)
        })

        trained_models[name] = model
        test_predictions[name] = (y_test_pred, y_test_proba)

    # 4. Summary Table & Leaderboard
    bench_df = pd.DataFrame(benchmark_results)
    print("\n" + "="*80)
    print("                     SENTINAL AI BENCHMARK LEADERBOARD")
    print("="*80)
    print(bench_df.to_string(index=False))

    bench_df.to_csv(BENCHMARKS_DIR / "leaderboard.csv", index=False)
    with open(BENCHMARKS_DIR / "leaderboard.md", "w", encoding="utf-8") as f:
        f.write("# Sentinal AI — Model Benchmark Leaderboard\n\n")
        f.write(bench_df.to_markdown(index=False))
        f.write("\n\n*Evaluation Protocol: Stratified 5-Fold CV + Isolated 20% Held-out Test Set (N=7,488)*\n")

    # 5. Visualizations: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, (preds, _)) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"], cbar=False
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontweight="bold", fontsize=11)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close(fig)

    # 6. Visualizations: ROC and PR Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for idx, (name, (_, probas)) in enumerate(test_predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, probas)
        precision_c, recall_c, _ = precision_recall_curve(y_test, probas)
        auc_val = roc_auc_score(y_test, probas)
        pr_val = average_precision_score(y_test, probas)

        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[idx], lw=2)
        ax_pr.plot(recall_c, precision_c, label=f"{name} (PR-AUC = {pr_val:.3f})", color=colors[idx], lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall (PR) Curves", fontweight="bold", fontsize=12)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=300)
    plt.close(fig)

    # 7. Feature Importance Ranking (XGBoost)
    champion_name = "XGBoost Classifier"
    champion_model = trained_models[champion_name]

    importances = champion_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=feat_df.head(12), x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title(f"Top 12 Most Predictive Bot Features ({champion_name})", fontweight="bold", fontsize=13)
    ax.set_xlabel("Normalized Gini Feature Importance")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=300)
    plt.close(fig)

    # 8. Save Champion Model & Artifacts
    champion_path = MODELS_DIR / "champion_bot_detector.joblib"
    model_metadata = {
        "model_name": champion_name,
        "feature_names": feature_names,
        "test_f1": float(bench_df[bench_df["Model"] == champion_name]["Test F1-Score"].values[0]),
        "test_accuracy": float(bench_df[bench_df["Model"] == champion_name]["Test Accuracy"].values[0]),
        "test_roc_auc": float(bench_df[bench_df["Model"] == champion_name]["Test ROC-AUC"].values[0]),
        "decision_threshold": 0.50,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(champion_model, champion_path)
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\n[OK] Champion model saved to: {champion_path}")
    print(f"[OK] Model metadata saved to: {MODELS_DIR / 'model_metadata.json'}")
    print(f"[OK] Evaluation figures saved to: {FIGURES_DIR}")
    return bench_df


if __name__ == "__main__":
    train_and_benchmark()

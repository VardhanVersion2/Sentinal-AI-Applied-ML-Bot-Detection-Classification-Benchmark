"""
Exploratory Data Analysis (EDA) Module for Sentinal AI.

Computes descriptive statistics, class comparisons (Bot vs. Human),
correlations, and exports publication-grade visual artifacts.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Set cache dir before importing matplotlib to prevent Windows sandbox permission errors
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "interim" / "clean_accounts.csv"


def run_eda(input_path: Path = CLEAN_DATA_FILE) -> dict:
    """
    Executes comprehensive EDA on the cleaned accounts dataset and outputs
    statistical comparisons and visual charts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"[*] Running EDA on {len(df):,} accounts...")

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Target Class Breakdown
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["account_type"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(counts.index.str.capitalize(), counts.values, color=colors, width=0.5, edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f"{yval:,} ({yval/len(df)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Ground-Truth Class Distribution (Human vs. Bot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Account Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Key Behavioral Feature Comparisons
    numeric_features = [
        "followers_count", "friends_count", "statuses_count",
        "favourites_count", "average_tweets_per_day", "account_age_days"
    ]

    summary_stats = df.groupby("is_bot")[numeric_features].agg(["mean", "median", "std"]).T
    print("\n=== COMPARATIVE NUMERICAL STATISTICS (0=Human, 1=Bot) ===")
    print(summary_stats.round(2))

    # 3. Log-Scale Behavioral Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(numeric_features):
        ax = axes[idx]
        log_vals = np.log1p(df[feature])
        sns.boxplot(
            data=df, x="account_type", y=log_vals, ax=ax,
            palette={"human": "#2ecc71", "bot": "#e74c3c"}, boxprops=dict(alpha=0.8)
        )
        ax.set_title(f"Log(1 + {feature})", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Log Scale")
        ax.set_xticklabels(["Human (0)", "Bot (1)"])

    plt.suptitle("Log-Transformed Account Behavioral Metrics by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "behavioral_boxplots.png", dpi=300)
    plt.close(fig)

    # 4. Reputation Ratio Analysis
    # Reputation = Followers / (Followers + Friends + 1)
    reputation = df["followers_count"] / (df["followers_count"] + df["friends_count"] + 1)
    df_rep = df.copy()
    df_rep["reputation_ratio"] = reputation

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=df_rep[df_rep["is_bot"] == 0]["reputation_ratio"],
        label="Human (Genuine)", color="#2ecc71", fill=True, alpha=0.4, ax=ax
    )
    sns.kdeplot(
        data=df_rep[df_rep["is_bot"] == 1]["reputation_ratio"],
        label="Bot (Automated)", color="#e74c3c", fill=True, alpha=0.4, ax=ax
    )
    ax.set_title("Reputation Ratio Density: Followers / (Followers + Friends + 1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reputation Ratio [0.0 = Pure Follower / Leech, 1.0 = Pure Influencer]")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "reputation_ratio_density.png", dpi=300)
    plt.close(fig)

    # 5. Correlation Matrix of Numerical Features with Target
    corr_features = numeric_features + ["default_profile", "default_profile_image", "geo_enabled", "verified", "is_bot"]
    corr_matrix = df[corr_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Features vs. is_bot)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

    print(f"\n[OK] Generated and saved 4 EDA figures to: {FIGURES_DIR}")
    return {"summary_stats": summary_stats}


if __name__ == "__main__":
    run_eda()

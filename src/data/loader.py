"""
Data Loader & Preprocessing Pipeline for Sentinal AI.

Ingests the raw 37,438 account benchmark dataset, cleans types, handles
nulls, validates labels, and outputs standardized interim tables.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "twitter_human_bots_dataset.csv"
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "clean_accounts.csv"


def load_and_clean_data(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    """
    Loads raw Twitter human-vs-bot accounts dataset, applies data sanitization,
    encodes binary target is_bot (0=Human, 1=Bot), and strips redundant index columns.

    Returns:
        pd.DataFrame: Sanitized accounts dataframe.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Run 'python src/data/download.py' first."
        )

    print(f"[*] Ingesting accounts from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 1. Drop redundant CSV index columns
    drop_cols = [c for c in ["Unnamed: 0", "index"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2. Construct binary target: 0 for human, 1 for bot
    if "account_type" not in df.columns:
        raise ValueError("Dataset is missing expected 'account_type' label column.")
    
    df["is_bot"] = (df["account_type"].str.lower().str.strip() == "bot").astype(int)

    # 3. Clean and normalize string fields
    text_cols = ["description", "location", "screen_name", "lang"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # 4. Standardize Boolean flags
    bool_cols = ["default_profile", "default_profile_image", "geo_enabled", "verified"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 5. Validate and enforce non-negative integers for count metrics
    count_cols = ["statuses_count", "followers_count", "friends_count", "favourites_count", "account_age_days"]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            df[col] = df[col].clip(lower=0)

    # 6. Validate average_tweets_per_day
    if "average_tweets_per_day" in df.columns:
        df["average_tweets_per_day"] = pd.to_numeric(
            df["average_tweets_per_day"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

    print(f"[OK] Cleaned {len(df):,} accounts.")
    print(f"[OK] Class distribution:\n{df['account_type'].value_counts().to_string()}")

    return df


def save_interim_data(df: pd.DataFrame, output_path: Path = INTERIM_FILE) -> Path:
    """Saves sanitized dataframe to interim storage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved interim cleaned data to: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_path


if __name__ == "__main__":
    df_clean = load_and_clean_data()
    save_interim_data(df_clean)

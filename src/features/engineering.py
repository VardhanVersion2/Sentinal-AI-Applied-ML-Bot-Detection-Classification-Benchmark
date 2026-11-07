"""
Feature Engineering Pipeline for Sentinal AI.

Constructs mathematically grounded, leak-free behavioral features,
log-scaled counters, bounded reputation ratios, and profile completeness
metrics from raw account metadata.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AccountFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature extraction transformer for Twitter accounts.
    Transforms raw tabular account metadata into clean, numerical feature vectors.
    """

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (stateless transformation, no target leakage)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input account dataframe into engineered feature matrix.
        Handles zero-divisions with Laplace smoothing and logs with log1p.
        """
        df = X.copy()
        features = pd.DataFrame(index=df.index)

        # Helper to safely retrieve column as Series
        def get_col(col_name, default_val):
            if col_name in df.columns:
                return df[col_name]
            return pd.Series(default_val, index=df.index)

        # 1. Numerical Count Features (Clip >= 0 to handle anomalies)
        followers = pd.to_numeric(get_col("followers_count", 0), errors="coerce").fillna(0).clip(lower=0)
        friends = pd.to_numeric(get_col("friends_count", 0), errors="coerce").fillna(0).clip(lower=0)
        statuses = pd.to_numeric(get_col("statuses_count", 0), errors="coerce").fillna(0).clip(lower=0)
        favourites = pd.to_numeric(get_col("favourites_count", 0), errors="coerce").fillna(0).clip(lower=0)
        age_days = pd.to_numeric(get_col("account_age_days", 1), errors="coerce").fillna(1).clip(lower=1)
        avg_tweets = pd.to_numeric(get_col("average_tweets_per_day", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

        # 2. Log-Transformed Counters (Addresses extreme skewness)
        features["log_followers"] = np.log1p(followers)
        features["log_friends"] = np.log1p(friends)
        features["log_statuses"] = np.log1p(statuses)
        features["log_favourites"] = np.log1p(favourites)
        features["log_account_age"] = np.log1p(age_days)

        # 3. Behavioral Ratios (Strictly zero-division safe)
        # Reputation Ratio: in [0, 1]. High = influencer/celebrity, Low = fake/leech
        features["reputation_ratio"] = followers / (followers + friends + 1.0)
        
        # Follow Ratio: Logarithmic ratio between following and followers
        features["follow_ratio"] = np.log1p(followers) / (np.log1p(friends) + 1.0)
        
        # Activity Intensity: posting pace relative to account age
        features["activity_intensity"] = statuses / (age_days + 1.0)
        
        # Engagement/Like Rate: favourites relative to posts
        features["like_to_tweet_ratio"] = favourites / (statuses + 1.0)
        
        features["average_tweets_per_day"] = avg_tweets

        # 4. Text & Handle Metadata Signals
        screen_names = get_col("screen_name", "").fillna("").astype(str)
        features["screen_name_length"] = screen_names.apply(len)
        features["screen_name_digits"] = screen_names.apply(lambda s: sum(c.isdigit() for c in s))
        features["screen_name_digit_ratio"] = features["screen_name_digits"] / features["screen_name_length"].clip(lower=1)

        descriptions = get_col("description", "").fillna("").astype(str)
        description_text = descriptions.str.lower()
        features["description_length"] = descriptions.apply(len)
        features["description_word_count"] = descriptions.apply(lambda s: len(s.split()))
        features["has_description"] = (features["description_length"] > 0).astype(int)
        features["has_url_in_desc"] = description_text.str.contains(r"https?://|www\.|t\.co", regex=True, na=False).astype(int)

        locations = get_col("location", "").fillna("").astype(str)
        location_text = locations.str.strip().str.lower()
        features["has_location"] = ((locations != "") & (~location_text.eq("unknown"))).astype(int)

        # 5. Profile Configuration Flags (Booleans converted to 0/1)
        def parse_bool_series(series):
            values = pd.Series(series, index=df.index, dtype="object")
            values = values.where(values.notna(), False)

            def normalize(value):
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return bool(value)
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "1", "yes", "y"}:
                        return True
                    if lowered in {"false", "0", "no", "n", ""}:
                        return False
                return bool(value) if value is not None else False

            return values.map(normalize).astype(int)

        features["default_profile"] = parse_bool_series(get_col("default_profile", False))
        features["default_profile_image"] = parse_bool_series(get_col("default_profile_image", False))
        features["geo_enabled"] = parse_bool_series(get_col("geo_enabled", False))
        features["verified"] = parse_bool_series(get_col("verified", False))

        # 6. Composite Profile Completeness Score
        features["profile_completeness"] = (
            features["has_description"]
            + features["has_location"]
            + (1 - features["default_profile"])
            + (1 - features["default_profile_image"])
            + features["geo_enabled"]
            + features["verified"]
        )

        # Verify no NaN or Inf in output matrix
        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self.feature_names_ = list(features.columns)

        return features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


def build_and_save_features(
    input_path: Path = Path("data/interim/clean_accounts.csv"),
    output_path: Path = Path("data/processed/features_matrix.csv"),
) -> pd.DataFrame:
    """Builds and saves the engineered feature matrix with target column."""
    df_clean = pd.read_csv(input_path)
    extractor = AccountFeatureExtractor()
    X_features = extractor.transform(df_clean)
    X_features["is_bot"] = df_clean["is_bot"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    X_features.to_csv(output_path, index=False)
    print(f"[OK] Engineered {X_features.shape[1]-1} features for {len(X_features):,} records.")
    print(f"[OK] Saved features matrix to: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    return X_features


if __name__ == "__main__":
    build_and_save_features()

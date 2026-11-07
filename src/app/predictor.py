"""
Inference Predictor Engine for Sentinal AI.

Loads the saved champion model and feature pipeline, applies input validation,
and returns structured predictions, probabilities, and risk tiers.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np

from src.features.engineering import AccountFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAMPION_MODEL_PATH = PROJECT_ROOT / "models" / "champion_bot_detector.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


class BotDetectorPredictor:
    """Production predictor wrapper for Sentinal AI."""

    def __init__(self, model_path: Path = CHAMPION_MODEL_PATH, meta_path: Path = METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
        
        self.model = joblib.load(model_path)
        self.feature_extractor = AccountFeatureExtractor()
        
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.threshold = self.metadata.get("decision_threshold", 0.50)
        self.feature_names = self.metadata.get("feature_names", [])

    def _determine_risk_tier(self, proba: float) -> str:
        """Classifies continuous bot probability into operational risk tiers."""
        if proba < 0.25:
            return "Low Risk (Genuine Human)"
        elif proba < 0.50:
            return "Moderate Risk (Likely Human)"
        elif proba < 0.75:
            return "High Risk (Probable Bot/Spammer)"
        else:
            return "Critical Risk (Automated Spambot)"

    def predict_single(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single Twitter account dictionary.

        Args:
            account_data (dict): Dictionary of account features (e.g. followers_count, screen_name).

        Returns:
            dict: Structured prediction output with label, probability, risk tier.
        """
        if not isinstance(account_data, dict):
            raise TypeError("account_data must be a dictionary of account metadata.")

        df_input = pd.DataFrame([account_data])
        X_features = self.feature_extractor.transform(df_input)

        # Align features to model's expected order
        if self.feature_names:
            for feat in self.feature_names:
                if feat not in X_features.columns:
                    X_features[feat] = 0.0
            X_features = X_features[self.feature_names]

        proba_bot = float(np.clip(self.model.predict_proba(X_features)[0, 1], 0.0, 1.0))
        is_bot = int(proba_bot >= self.threshold)

        return {
            "is_bot": is_bot,
            "prediction_label": "Bot" if is_bot == 1 else "Human",
            "bot_probability": round(proba_bot, 4),
            "confidence_score": round(proba_bot if is_bot == 1 else (1.0 - proba_bot), 4),
            "risk_tier": self._determine_risk_tier(proba_bot),
            "decision_threshold": self.threshold
        }

    def predict_batch(self, accounts_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vectorized batch prediction for multiple accounts."""
        if not accounts_list:
            return []

        if not isinstance(accounts_list, list) or any(not isinstance(item, dict) for item in accounts_list):
            raise TypeError("accounts_list must be a list of dictionaries, one per account.")

        df_input = pd.DataFrame(accounts_list)
        X_features = self.feature_extractor.transform(df_input)

        if self.feature_names:
            for feat in self.feature_names:
                if feat not in X_features.columns:
                    X_features[feat] = 0.0
            X_features = X_features[self.feature_names]

        probas_bot = self.model.predict_proba(X_features)[:, 1]
        
        results = []
        for i, proba in enumerate(probas_bot):
            p = float(np.clip(proba, 0.0, 1.0))
            is_b = int(p >= self.threshold)
            results.append({
                "account_index": i,
                "is_bot": is_b,
                "prediction_label": "Bot" if is_b == 1 else "Human",
                "bot_probability": round(p, 4),
                "risk_tier": self._determine_risk_tier(p)
            })

        return results

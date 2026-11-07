"""
Unit and Integration Test Suite for Sentinal AI.

Tests:
1. Feature extraction & zero-division handling
2. Inference predictor consistency and probability bounds
3. Flask API endpoints (/health, /predict, /predict_batch, error cases)
"""

import unittest
import json
import pandas as pd
import numpy as np

from src.features.engineering import AccountFeatureExtractor
from src.app.predictor import BotDetectorPredictor
from src.app.api import app


class TestFeatureEngineering(unittest.TestCase):
    """Tests the AccountFeatureExtractor pipeline."""

    def setUp(self):
        self.extractor = AccountFeatureExtractor()

    def test_zero_division_safety(self):
        """Verifies that 0 followers and 0 friends do not trigger ZeroDivisionError."""
        df_edge = pd.DataFrame([{
            "followers_count": 0,
            "friends_count": 0,
            "statuses_count": 0,
            "favourites_count": 0,
            "account_age_days": 0,
            "screen_name": "",
            "description": ""
        }])
        features = self.extractor.transform(df_edge)
        self.assertEqual(features["reputation_ratio"].iloc[0], 0.0)
        self.assertFalse(features.isna().any().any())
        self.assertFalse(np.isinf(features.values).any())

    def test_screen_name_digit_extraction(self):
        """Verifies that digits and length are correctly calculated."""
        df = pd.DataFrame([{
            "screen_name": "bot998877",
            "description": "Visit http://spam.xyz"
        }])
        features = self.extractor.transform(df)
        self.assertEqual(features["screen_name_digits"].iloc[0], 6)
        self.assertEqual(features["screen_name_length"].iloc[0], 9)
        self.assertEqual(features["has_url_in_desc"].iloc[0], 1)

    def test_string_boolean_and_uppercase_url_handling(self):
        """Verifies boolean-like strings and uppercase URLs are normalized safely."""
        df = pd.DataFrame([{
            "screen_name": "BOT_42",
            "description": "FREE BITCOIN AIRDROP! HTTP://FAKE.SCAM",
            "default_profile": "true",
            "default_profile_image": "False",
            "geo_enabled": "1",
            "verified": "0",
            "location": "UNKNOWN"
        }])
        features = self.extractor.transform(df)
        self.assertEqual(features["has_url_in_desc"].iloc[0], 1)
        self.assertEqual(features["default_profile"].iloc[0], 1)
        self.assertEqual(features["default_profile_image"].iloc[0], 0)
        self.assertEqual(features["geo_enabled"].iloc[0], 1)
        self.assertEqual(features["verified"].iloc[0], 0)
        self.assertEqual(features["has_location"].iloc[0], 0)


class TestPredictor(unittest.TestCase):
    """Tests the champion model inference wrapper."""

    def setUp(self):
        self.predictor = BotDetectorPredictor()

    def test_single_prediction_human_profile(self):
        """Tests inference on a high-reputation human profile."""
        human_account = {
            "screen_name": "john_doe_researcher",
            "followers_count": 4500,
            "friends_count": 600,
            "statuses_count": 12500,
            "favourites_count": 8900,
            "account_age_days": 2800,
            "default_profile": False,
            "default_profile_image": False,
            "geo_enabled": True,
            "verified": True,
            "description": "Senior ML Scientist at AI Lab. Working on reproducible systems."
        }
        res = self.predictor.predict_single(human_account)
        self.assertIn("is_bot", res)
        self.assertIn("bot_probability", res)
        self.assertIn("risk_tier", res)
        self.assertTrue(0.0 <= res["bot_probability"] <= 1.0)
        self.assertEqual(res["is_bot"], 0)
        self.assertEqual(res["prediction_label"], "Human")

    def test_single_prediction_spambot_profile(self):
        """Tests inference on a typical throwaway spambot profile."""
        bot_account = {
            "screen_name": "free_crypto_984321",
            "followers_count": 2,
            "friends_count": 850,
            "statuses_count": 45,
            "favourites_count": 0,
            "account_age_days": 12,
            "default_profile": True,
            "default_profile_image": True,
            "geo_enabled": False,
            "verified": False,
            "description": "FREE BITCOIN AIRDROP! CLICK LINK BELOW http://fake.scam"
        }
        res = self.predictor.predict_single(bot_account)
        self.assertEqual(res["is_bot"], 1)
        self.assertEqual(res["prediction_label"], "Bot")
        self.assertGreater(res["bot_probability"], 0.70)

    def test_invalid_account_type_raises_error(self):
        """Verifies invalid direct predictor input fails with a clear error."""
        with self.assertRaises(TypeError):
            self.predictor.predict_single(None)


class TestFlaskAPI(unittest.TestCase):
    """Tests the Flask REST API endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        """GET /health returns 200 and valid metadata."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_accuracy", data)

    def test_predict_endpoint_valid(self):
        """POST /predict with valid account returns classification result."""
        payload = {
            "screen_name": "test_account_42",
            "followers_count": 1000,
            "friends_count": 500,
            "statuses_count": 4000,
            "favourites_count": 2000,
            "account_age_days": 1200,
            "default_profile": False,
            "default_profile_image": False,
            "geo_enabled": False,
            "verified": False,
            "description": "Just an everyday Twitter user."
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("is_bot", data)
        self.assertIn("latency_ms", data)

    def test_predict_endpoint_invalid_payload(self):
        """POST /predict with invalid non-JSON returns 400 error."""
        resp = self.client.post("/predict", data="not json", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_predict_batch_endpoint(self):
        """POST /predict_batch with multiple accounts returns vectorized results."""
        payload = {
            "accounts": [
                {"followers_count": 5000, "friends_count": 400, "screen_name": "human1"},
                {"followers_count": 1, "friends_count": 900, "default_profile": True, "screen_name": "bot999"}
            ]
        }
        resp = self.client.post("/predict_batch", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["total_accounts"], 2)
        self.assertEqual(len(data["results"]), 2)


if __name__ == "__main__":
    unittest.main()

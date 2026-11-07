"""
Production Flask Prediction API for Sentinal AI.

Exposes RESTful endpoints for real-time and batch bot detection inference.
"""

import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from src.app.predictor import BotDetectorPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
APP_START_TIME = time.time()

# Initialize predictor singleton
try:
    predictor = BotDetectorPredictor()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    init_error = str(e)


@app.route("/", methods=["GET"])
def index():
    """Serve the interactive account analysis dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint exposing system uptime, model status, and metadata."""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model failed to load",
            "error": init_error
        }), 500

    return jsonify({
        "status": "healthy",
        "service": "Sentinal AI — Twitter Bot Detection API",
        "model_name": predictor.metadata.get("model_name", "Unknown"),
        "model_accuracy": predictor.metadata.get("test_accuracy"),
        "model_roc_auc": predictor.metadata.get("test_roc_auc"),
        "uptime_seconds": round(time.time() - APP_START_TIME, 2),
        "features_count": len(predictor.feature_names)
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint for a single account.
    
    Expected JSON payload:
    {
        "screen_name": "example_user",
        "followers_count": 1200,
        "friends_count": 450,
        "statuses_count": 8900,
        "favourites_count": 3400,
        "account_age_days": 1500,
        "default_profile": false,
        "default_profile_image": false,
        "verified": false,
        "geo_enabled": true,
        "description": "Tech enthusiast & software engineer"
    }
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({
            "error": "Invalid request body. Expected a JSON object with account metadata."
        }), 400

    try:
        start_time = time.time()
        result = predictor.predict_single(payload)
        result["latency_ms"] = round((time.time() - start_time) * 1000, 3)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Vectorized batch inference endpoint.
    
    Expected JSON payload:
    {
        "accounts": [ {...}, {...} ]
    }
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict) or "accounts" not in payload:
        return jsonify({
            "error": "Invalid request body. Expected a JSON object with an 'accounts' array."
        }), 400

    accounts = payload["accounts"]
    if not isinstance(accounts, list) or len(accounts) == 0:
        return jsonify({"error": "'accounts' must be a non-empty list of account dictionaries."}), 400

    if len(accounts) > 1000:
        return jsonify({"error": "Batch size exceeds maximum limit of 1,000 accounts per request."}), 413

    try:
        start_time = time.time()
        batch_results = predictor.predict_batch(accounts)
        total_latency = (time.time() - start_time) * 1000
        return jsonify({
            "total_accounts": len(batch_results),
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_per_sample_ms": round(total_latency / len(batch_results), 4),
            "results": batch_results
        }), 200
    except Exception as e:
        return jsonify({"error": f"Batch inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

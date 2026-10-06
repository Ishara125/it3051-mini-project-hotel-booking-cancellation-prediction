"""Flask Application for Hotel Booking Cancellation Prediction.

Provides the web user interface and REST endpoints for prediction and health checks.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.inference import get_pipeline
    from app.validation import ValidationError
except ImportError:
    from inference import get_pipeline
    from validation import ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_CASES_PATH = BASE_DIR / "results" / "system" / "demo_test_cases.json"

app = Flask(__name__)


# Eagerly load the model pipeline at startup to verify integrity
try:
    pipeline = get_pipeline()
    logger.info("Pipeline ready for serving predictions.")
except Exception as e:
    logger.error("Failed to initialize inference pipeline at startup: %s", e)
    raise


@app.route("/", methods=["GET"])
def index():
    """Serves the landing page."""
    return render_template("landing.html")


@app.route("/predictor", methods=["GET"])
def predictor():
    """Serves the prediction form interface."""
    return render_template("predictor.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming model status."""
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline.model is not None,
        "model_name": "Gradient Boosting Classifier (Tuned)",
        "expected_features": 899,
        "leakage_excluded": ["reservation_status", "reservation_status_date"],
        "target_excluded": "is_canceled"
    }), 200


@app.route("/api/demo-cases", methods=["GET"])
def get_demo_cases():
    """Returns verified demo test cases for live demonstrations."""
    if DEMO_CASES_PATH.exists():
        with open(DEMO_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200
    return jsonify([]), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict cancellation risk for a hotel booking.

    Accepts JSON payloads or standard HTML form submissions.
    """
    try:
        if request.is_json:
            payload = request.get_json()
        elif request.form:
            payload = request.form.to_dict()
        else:
            return jsonify({
                "status": "error",
                "message": "Empty or unsupported request payload. Please send JSON or form data."
            }), 400

        result = pipeline.predict(payload)

        # If AJAX / JSON client:
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
            return jsonify({
                "status": "success",
                "data": result
            }), 200

        # If traditional form submission:
        return render_template("predictor.html", result=result, form_data=payload)

    except ValidationError as val_err:
        logger.warning("Validation failed: %s", val_err.message)
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
            return jsonify({
                "status": "error",
                "message": val_err.message,
                "errors": val_err.errors
            }), 400
        return render_template("predictor.html", error=val_err.message, form_data=request.form.to_dict()), 400

    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        user_message = "An error occurred while evaluating the booking. Please check your inputs and try again."
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
            return jsonify({
                "status": "error",
                "message": user_message
            }), 500
        return render_template("predictor.html", error=user_message, form_data=request.form.to_dict() if request.form else {}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error."}), 500


if __name__ == "__main__":
    logger.info("Starting Hotel Booking Cancellation Prediction Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

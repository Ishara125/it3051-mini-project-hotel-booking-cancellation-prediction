"""Safe Inference Pipeline for Hotel Booking Cancellation Prediction.

This module loads the ALREADY TRAINED production artifacts once at startup.
No model retraining (fit) is ever executed here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from app.validation import validate_booking_input, ValidationError
except ImportError:
    from validation import validate_booking_input, ValidationError

logger = logging.getLogger(__name__)

# Base directory resolved relative to repository root (portable across machines)
BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "results" / "modeling" / "final_model_bundle.joblib"
MODEL_PATH = BASE_DIR / "results" / "modeling" / "final_gradient_boosting_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "results" / "modeling" / "final_preprocessor.joblib"
METADATA_PATH = BASE_DIR / "results" / "modeling" / "final_model_metadata.json"

EXPECTED_TRANSFORMED_FEATURES = 899


class PredictionPipeline:
    """Encapsulates the trained model, preprocessor, and inference transformations."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.numerical_features: list[str] = []
        self.categorical_features: list[str] = []
        self.expected_columns: list[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads and verifies pre-trained artifacts without any retraining."""
        if BUNDLE_PATH.exists():
            logger.info("Loading production artifacts from bundle: %s", BUNDLE_PATH)
            try:
                bundle = joblib.load(BUNDLE_PATH)
                self.model = bundle["model"]
                self.preprocessor = bundle["preprocessor"]
                self.numerical_features = list(bundle["numerical_features"])
                self.categorical_features = list(bundle["categorical_features"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load model bundle from '{BUNDLE_PATH}': {exc}"
                ) from exc
        elif MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
            logger.info("Loading separate model and preprocessor artifacts...")
            try:
                self.model = joblib.load(MODEL_PATH)
                self.preprocessor = joblib.load(PREPROCESSOR_PATH)
                # Extract columns from ColumnTransformer
                for name, trans, cols in self.preprocessor.transformers_:
                    if name == "num":
                        self.numerical_features = list(cols)
                    elif name == "cat":
                        self.categorical_features = list(cols)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load standalone model artifacts: {exc}"
                ) from exc
        else:
            raise FileNotFoundError(
                f"Missing required model artifacts! Checked {BUNDLE_PATH} and "
                f"({MODEL_PATH}, {PREPROCESSOR_PATH})."
            )

        # Verification checks
        if not hasattr(self.model, "predict") or not hasattr(self.model, "predict_proba"):
            raise TypeError("Loaded model does not expose predict() and predict_proba().")

        if not hasattr(self.preprocessor, "transform"):
            raise TypeError("Loaded preprocessor does not expose transform().")

        self.expected_columns = self.numerical_features + self.categorical_features
        if len(self.expected_columns) != 37:
            raise ValueError(
                f"Expected 37 predictor features for preprocessor, found {len(self.expected_columns)}."
            )

        logger.info(
            "Model artifacts loaded successfully. Expected transformed feature count: %d",
            EXPECTED_TRANSFORMED_FEATURES,
        )

    def engineer_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Applies deterministic, leak-free feature engineering for single or batch bookings.

        Reproduces the exact logic of src/feature_engineering/feature_engineering.py:
        1. total_guests = adults + children + babies
        2. total_stay = stays_in_week_nights + stays_in_weekend_nights
        3. is_family = 1 if (children + babies > 0) else 0
        4. room_changed = 1 if (reserved_room_type != assigned_room_type) else 0
        5. has_special_requests = 1 if (total_of_special_requests > 0) else 0
        6. has_previous_cancellations = 1 if (previous_cancellations > 0) else 0
        7. has_previous_bookings = 1 if (previous_bookings_not_canceled > 0) else 0
        8. is_weekend_only = 1 if (stays_in_weekend_nights > 0 and stays_in_week_nights == 0) else 0
        """
        df = df_raw.copy()

        # Handle missing children safely
        children_val = df["children"].fillna(0)
        df["total_guests"] = (df["adults"] + children_val + df["babies"]).astype(int)
        df["total_stay"] = (df["stays_in_week_nights"] + df["stays_in_weekend_nights"]).astype(int)
        df["is_family"] = ((children_val + df["babies"]) > 0).astype(int)
        df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
        df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)
        df["has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
        df["has_previous_bookings"] = (df["previous_bookings_not_canceled"] > 0).astype(int)
        df["is_weekend_only"] = (
            (df["stays_in_weekend_nights"] > 0) & (df["stays_in_week_nights"] == 0)
        ).astype(int)

        return df

    def predict(self, raw_input_dict: dict[str, Any]) -> dict[str, Any]:
        """End-to-end safe prediction for a single raw booking input.

        Validation -> One-row DataFrame -> Feature Engineering -> Column Alignment ->
        Preprocessor.transform() -> Model.predict_proba() -> Response Formatting.
        """
        # 1. Validation
        cleaned = validate_booking_input(raw_input_dict)

        # 2. One-row DataFrame
        df_single = pd.DataFrame([cleaned])

        # 3. Deterministic Feature Engineering
        df_engineered = self.engineer_features(df_single)

        # 4. Ensure exact expected feature columns and order
        missing = [c for c in self.expected_columns if c not in df_engineered.columns]
        if missing:
            raise ValueError(f"Missing required columns for preprocessor: {missing}")

        X_input = df_engineered[self.expected_columns].copy()

        # 5. Preprocessor transformation (NEVER FIT)
        X_processed = self.preprocessor.transform(X_input)

        if X_processed.shape[1] != EXPECTED_TRANSFORMED_FEATURES:
            raise AssertionError(
                f"Preprocessor produced {X_processed.shape[1]} features, "
                f"expected {EXPECTED_TRANSFORMED_FEATURES}."
            )

        # 6. Model prediction (NEVER RETRAIN)
        pred_class = int(self.model.predict(X_processed)[0])
        probabilities = self.model.predict_proba(X_processed)[0]
        prob_not_cancel = float(probabilities[0])
        prob_cancel = float(probabilities[1])

        # 7. UI risk categorization
        if prob_cancel >= 0.65:
            risk_level = "High"
            risk_badge = "risk-high"
        elif prob_cancel >= 0.35:
            risk_level = "Medium"
            risk_badge = "risk-medium"
        else:
            risk_level = "Low"
            risk_badge = "risk-low"

        return {
            "prediction": pred_class,
            "prediction_label": "Likely to Cancel" if pred_class == 1 else "Likely Not to Cancel",
            "cancellation_probability": round(prob_cancel, 4),
            "cancellation_probability_pct": f"{prob_cancel * 100:.1f}%",
            "non_cancellation_probability": round(prob_not_cancel, 4),
            "non_cancellation_probability_pct": f"{prob_not_cancel * 100:.1f}%",
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "transformed_features_count": int(X_processed.shape[1]),
            "note": (
                "This prediction is based on historical booking patterns and should support, "
                "not replace, business decision-making."
            ),
        }


# Global singleton instance loaded once at startup
_pipeline_instance: PredictionPipeline | None = None


def get_pipeline() -> PredictionPipeline:
    """Returns the loaded pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PredictionPipeline()
    return _pipeline_instance

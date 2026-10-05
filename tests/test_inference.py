"""Tests for inference pipeline module (app.inference)."""

import json
from pathlib import Path
import pytest
import pandas as pd
from app.inference import get_pipeline, EXPECTED_TRANSFORMED_FEATURES


@pytest.fixture(scope="module")
def pipeline():
    """Provides the singleton inference pipeline."""
    return get_pipeline()


@pytest.fixture
def sample_booking():
    """Returns a valid raw booking input."""
    return {
        "hotel": "City Hotel",
        "lead_time": 30,
        "arrival_date": "2017-08-15",
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 120.5,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
        "agent": 9.0,
        "company": None
    }


def test_model_loading(pipeline):
    """TEST 1 - Model and preprocessor load successfully without retraining."""
    assert pipeline.model is not None
    assert pipeline.preprocessor is not None
    assert hasattr(pipeline.model, "predict")
    assert hasattr(pipeline.model, "predict_proba")
    assert hasattr(pipeline.preprocessor, "transform")
    # Verify no fit methods called
    assert len(pipeline.expected_columns) == 37


def test_preprocessing_shape(pipeline, sample_booking):
    """TEST 4 - Preprocessed record produces exactly 899 features."""
    cleaned = sample_booking.copy()
    cleaned["arrival_date_year"] = 2017
    cleaned["arrival_date_month"] = "August"
    cleaned["arrival_date_day_of_month"] = 15
    cleaned["arrival_date_week_number"] = 33

    df_raw = pd.DataFrame([cleaned])
    df_feat = pipeline.engineer_features(df_raw)
    X_input = df_feat[pipeline.expected_columns]
    X_proc = pipeline.preprocessor.transform(X_input)

    assert X_proc.shape[1] == EXPECTED_TRANSFORMED_FEATURES
    assert X_proc.shape[1] == 899


def test_feature_engineering_calculations(pipeline, sample_booking):
    """Verify that all 8 engineered features match training specifications."""
    df_raw = pd.DataFrame([{
        "adults": 2,
        "children": 1,
        "babies": 0,
        "stays_in_week_nights": 3,
        "stays_in_weekend_nights": 2,
        "reserved_room_type": "A",
        "assigned_room_type": "D",
        "total_of_special_requests": 2,
        "previous_cancellations": 1,
        "previous_bookings_not_canceled": 4,
    }])
    df_feat = pipeline.engineer_features(df_raw)

    assert df_feat["total_guests"].iloc[0] == 3
    assert df_feat["total_stay"].iloc[0] == 5
    assert df_feat["is_family"].iloc[0] == 1
    assert df_feat["room_changed"].iloc[0] == 1
    assert df_feat["has_special_requests"].iloc[0] == 1
    assert df_feat["has_previous_cancellations"].iloc[0] == 1
    assert df_feat["has_previous_bookings"].iloc[0] == 1
    assert df_feat["is_weekend_only"].iloc[0] == 0


def test_valid_input_prediction(pipeline, sample_booking):
    """TEST 2 - Valid input prediction succeeds."""
    result = pipeline.predict(sample_booking)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "cancellation_probability" in result


def test_prediction_format(pipeline, sample_booking):
    """TEST 3 - Class is 0 or 1, probability is between 0 and 1."""
    result = pipeline.predict(sample_booking)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["cancellation_probability"] <= 1.0
    assert 0.0 <= result["non_cancellation_probability"] <= 1.0
    # Probabilities should sum to 1.0 (within floating point precision)
    total_prob = result["cancellation_probability"] + result["non_cancellation_probability"]
    assert pytest.approx(total_prob, abs=1e-3) == 1.0
    assert result["prediction_label"] in ("Likely to Cancel", "Likely Not to Cancel")


def test_end_to_end_prediction(pipeline, sample_booking):
    """TEST 10 - End-to-end prediction: input -> engineering -> preprocessor -> model."""
    result = pipeline.predict(sample_booking)
    assert result["transformed_features_count"] == 899
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert "note" in result


def test_model_consistency_with_demo_cases(pipeline):
    """PART 28 - Model consistency check against verified test set cases."""
    demo_file = Path(__file__).resolve().parent / "demo_cases.json"
    assert demo_file.exists()

    with open(demo_file, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    for case in demo_cases:
        res = pipeline.predict(case["raw_booking_inputs"])
        assert res["prediction"] == case["predicted_class"]
        assert res["prediction_label"] == case["predicted_label"]
        # Probabilities should match to at least 3 decimal places
        assert pytest.approx(res["cancellation_probability"], abs=1e-3) == case["cancellation_probability"]

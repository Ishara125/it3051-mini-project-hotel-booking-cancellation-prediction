"""Tests for Flask backend endpoints (app.app)."""

import pytest
from app.app import app


@pytest.fixture
def client():
    """Provides a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_payload():
    """Baseline valid payload for API requests."""
    return {
        "hotel": "City Hotel",
        "lead_time": 45,
        "arrival_date": "2017-09-10",
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "GBR",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 98.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1
    }


def test_homepage_serves_html(client):
    """GET / serves the landing page with key content."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Hotel" in html
    assert "Cancellation" in html


def test_predictor_page_serves_html(client):
    """GET /predictor serves the prediction form page."""
    response = client.get("/predictor")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "bookingForm" in html
    assert "Hotel Booking Cancellation Prediction" in html


def test_health_endpoint(client):
    """TEST 9 - GET /health returns status ok, model_loaded: true."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["expected_features"] == 899
    assert "reservation_status" in data["leakage_excluded"]


def test_demo_cases_api(client):
    """GET /api/demo-cases returns pre-calculated demo test cases."""
    response = client.get("/api/demo-cases")
    assert response.status_code == 200
    cases = response.get_json()
    assert isinstance(cases, list)
    assert len(cases) >= 3


def test_predict_endpoint_valid_json(client, valid_payload):
    """POST /predict with valid JSON returns 200 and prediction data."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    res = data["data"]
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["cancellation_probability"] <= 1.0
    assert res["transformed_features_count"] == 899


def test_predict_endpoint_empty_request(client):
    """TEST 8 - Empty request returns 400-level error, not application crash."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "message" in data


def test_predict_endpoint_invalid_negative_number(client, valid_payload):
    """POST /predict with negative lead_time returns 400 with descriptive error."""
    valid_payload["lead_time"] = -25
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "lead_time" in data["errors"]


def test_predict_endpoint_invalid_category(client, valid_payload):
    """POST /predict with invalid hotel category returns 400."""
    valid_payload["hotel"] = "Underwater Hotel"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "hotel" in data["errors"]


def test_predict_endpoint_leakage_rejection(client, valid_payload):
    """POST /predict rejecting leakage columns."""
    valid_payload["reservation_status"] = "Canceled"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "reservation_status" in data["errors"]

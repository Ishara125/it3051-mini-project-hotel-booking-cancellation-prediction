"""Tests for input validation module (app.validation)."""

import pytest
from app.validation import validate_booking_input, ValidationError


@pytest.fixture
def valid_booking():
    """Returns a baseline valid booking dictionary."""
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


def test_valid_input_passes(valid_booking):
    """Test that a complete valid input passes validation and converts types."""
    cleaned = validate_booking_input(valid_booking)
    assert cleaned["hotel"] == "City Hotel"
    assert cleaned["lead_time"] == 30
    assert cleaned["arrival_date_year"] == 2017
    assert cleaned["arrival_date_month"] == "August"
    assert cleaned["arrival_date_day_of_month"] == 15
    assert cleaned["arrival_date_week_number"] == 33
    assert cleaned["adults"] == 2
    assert cleaned["adr"] == 120.5


def test_missing_required_input(valid_booking):
    """TEST 5 - Missing required input yields clear validation error."""
    del valid_booking["lead_time"]
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "lead_time" in exc_info.value.errors


def test_negative_impossible_number(valid_booking):
    """TEST 6 - Negative impossible number yields validation error."""
    valid_booking["lead_time"] = -10
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "lead_time" in exc_info.value.errors
    assert "at least 0" in exc_info.value.errors["lead_time"]


def test_negative_adr(valid_booking):
    """Test that negative ADR is rejected based on cleaned data rules."""
    valid_booking["adr"] = -5.0
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "adr" in exc_info.value.errors


def test_extreme_adr(valid_booking):
    """Test that extreme ADR > 5000 is rejected based on cleaned data rules."""
    valid_booking["adr"] = 6000.0
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "adr" in exc_info.value.errors


def test_zero_guests_rejected(valid_booking):
    """Test that 0 total guests is rejected based on cleaned data rules."""
    valid_booking["adults"] = 0
    valid_booking["children"] = 0
    valid_booking["babies"] = 0
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "guests" in exc_info.value.errors


def test_invalid_category(valid_booking):
    """TEST 7 - Invalid category yields clear validation response."""
    valid_booking["hotel"] = "Space Hotel"
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "hotel" in exc_info.value.errors


def test_invalid_market_segment(valid_booking):
    """Test invalid market segment rejection."""
    valid_booking["market_segment"] = "InvalidSegment"
    with pytest.raises(ValidationError) as exc_info:
        validate_booking_input(valid_booking)
    assert "market_segment" in exc_info.value.errors


def test_prohibited_leakage_columns_rejected(valid_booking):
    """Test that target and post-outcome leakage columns are strictly rejected."""
    for leak_col in ["reservation_status", "reservation_status_date", "is_canceled"]:
        bad_input = dict(valid_booking)
        bad_input[leak_col] = "Check-Out"
        with pytest.raises(ValidationError) as exc_info:
            validate_booking_input(bad_input)
        assert leak_col in exc_info.value.errors


def test_empty_request():
    """TEST 8 (part 1) - Empty request raises validation error."""
    with pytest.raises(ValidationError):
        validate_booking_input({})

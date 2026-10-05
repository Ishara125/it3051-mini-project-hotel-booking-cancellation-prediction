"""Input Validation Module for Hotel Booking Cancellation Prediction.

Enforces domain constraints, datatype verification, and leakage prevention
derived from the cleaned project dataset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Allowed categorical sets verified against the training preprocessor
VALID_HOTELS = {"City Hotel", "Resort Hotel"}
VALID_MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
}
VALID_MEALS = {"BB", "FB", "HB", "SC", "Undefined"}
VALID_MARKET_SEGMENTS = {
    "Online TA", "Offline TA/TO", "Direct", "Corporate",
    "Groups", "Complementary", "Aviation", "Undefined"
}
VALID_DISTRIBUTION_CHANNELS = {"TA/TO", "Direct", "Corporate", "GDS", "Undefined"}
VALID_RESERVED_ROOMS = {"A", "B", "C", "D", "E", "F", "G", "H", "L"}
VALID_ASSIGNED_ROOMS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "K"}
VALID_DEPOSIT_TYPES = {"No Deposit", "Non Refund", "Refundable"}
VALID_CUSTOMER_TYPES = {"Transient", "Transient-Party", "Contract", "Group"}

PROHIBITED_LEAKAGE_COLUMNS = {"reservation_status", "reservation_status_date", "is_canceled"}


class ValidationError(Exception):
    """Raised when user input fails domain validation."""
    def __init__(self, message: str, errors: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or {"general": message}


def _parse_int(value: Any, field_name: str, min_val: int = 0, max_val: int | None = None) -> int:
    """Helper to safely parse and validate an integer field."""
    if value is None or str(value).strip() == "":
        raise ValidationError(f"Field '{field_name}' is required.", {field_name: "Required field."})
    try:
        val = int(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Field '{field_name}' must be an integer, got '{value}'.",
            {field_name: "Must be a valid integer."}
        )
    if val < min_val:
        raise ValidationError(
            f"Field '{field_name}' cannot be less than {min_val}, got {val}.",
            {field_name: f"Must be at least {min_val}."}
        )
    if max_val is not None and val > max_val:
        raise ValidationError(
            f"Field '{field_name}' cannot exceed {max_val}, got {val}.",
            {field_name: f"Cannot exceed {max_val}."}
        )
    return val


def _parse_float(value: Any, field_name: str, min_val: float = 0.0, max_val: float | None = None) -> float:
    """Helper to safely parse and validate a float field."""
    if value is None or str(value).strip() == "":
        raise ValidationError(f"Field '{field_name}' is required.", {field_name: "Required field."})
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Field '{field_name}' must be a numeric value, got '{value}'.",
            {field_name: "Must be a valid number."}
        )
    if val < min_val:
        raise ValidationError(
            f"Field '{field_name}' cannot be less than {min_val}, got {val}.",
            {field_name: f"Must be at least {min_val}."}
        )
    if max_val is not None and val > max_val:
        raise ValidationError(
            f"Field '{field_name}' cannot exceed {max_val}, got {val}.",
            {field_name: f"Cannot exceed {max_val}."}
        )
    return val


def validate_booking_input(raw_data: dict[str, Any]) -> dict[str, Any]:
    """Validates raw user input against domain rules and returns cleaned data.

    Parameters
    ----------
    raw_data : dict[str, Any]
        Dictionary of raw booking attributes from web form or API.

    Returns
    -------
    dict[str, Any]
        Cleaned, type-converted dictionary ready for feature engineering.

    Raises
    ------
    ValidationError
        If any validation rule fails.
    """
    if not isinstance(raw_data, dict):
        raise ValidationError("Request payload must be a JSON object or dictionary.", {"payload": "Invalid payload format."})

    errors: dict[str, str] = {}

    # 1. Prohibit leakage and target fields
    for leak_col in PROHIBITED_LEAKAGE_COLUMNS:
        if leak_col in raw_data:
            errors[leak_col] = (
                f"Prohibited field '{leak_col}' detected. Leakage columns and the "
                "target variable must never be provided as inputs."
            )

    if errors:
        raise ValidationError("Prohibited target or leakage fields in input.", errors)

    cleaned: dict[str, Any] = {}

    # 2. Hotel
    hotel = str(raw_data.get("hotel", "")).strip()
    if hotel not in VALID_HOTELS:
        errors["hotel"] = f"Hotel must be one of: {sorted(VALID_HOTELS)}"
    else:
        cleaned["hotel"] = hotel

    # 3. Lead Time
    try:
        cleaned["lead_time"] = _parse_int(raw_data.get("lead_time"), "lead_time", min_val=0, max_val=1000)
    except ValidationError as err:
        errors.update(err.errors)

    # 4. Arrival Date handling
    # Accepts either an ISO arrival_date (YYYY-MM-DD) or separate year/month/day/week
    arrival_date_str = raw_data.get("arrival_date")
    if arrival_date_str and str(arrival_date_str).strip():
        try:
            dt = datetime.strptime(str(arrival_date_str).strip(), "%Y-%m-%d")
            cleaned["arrival_date_year"] = dt.year
            cleaned["arrival_date_month"] = dt.strftime("%B")
            cleaned["arrival_date_day_of_month"] = dt.day
            cleaned["arrival_date_week_number"] = dt.isocalendar()[1]
        except ValueError:
            errors["arrival_date"] = "Arrival date must be in YYYY-MM-DD format."
    else:
        # Separate fields
        try:
            cleaned["arrival_date_year"] = _parse_int(
                raw_data.get("arrival_date_year", 2017), "arrival_date_year", min_val=2015, max_val=2035
            )
        except ValidationError as err:
            errors.update(err.errors)

        month = str(raw_data.get("arrival_date_month", "")).strip().capitalize()
        if month not in VALID_MONTHS:
            errors["arrival_date_month"] = f"Month must be one of: {sorted(VALID_MONTHS)}"
        else:
            cleaned["arrival_date_month"] = month

        try:
            cleaned["arrival_date_day_of_month"] = _parse_int(
                raw_data.get("arrival_date_day_of_month"), "arrival_date_day_of_month", min_val=1, max_val=31
            )
        except ValidationError as err:
            errors.update(err.errors)

        # Optional or computed week number
        if "arrival_date_week_number" in raw_data and raw_data.get("arrival_date_week_number") not in (None, ""):
            try:
                cleaned["arrival_date_week_number"] = _parse_int(
                    raw_data.get("arrival_date_week_number"), "arrival_date_week_number", min_val=1, max_val=53
                )
            except ValidationError as err:
                errors.update(err.errors)
        else:
            # Estimate week number from month and day if valid
            try:
                m_num = datetime.strptime(month, "%B").month
                est_dt = datetime(cleaned.get("arrival_date_year", 2017), m_num, min(cleaned.get("arrival_date_day_of_month", 15), 28))
                cleaned["arrival_date_week_number"] = est_dt.isocalendar()[1]
            except Exception:
                cleaned["arrival_date_week_number"] = 28

    # 5. Stay Nights
    try:
        cleaned["stays_in_weekend_nights"] = _parse_int(
            raw_data.get("stays_in_weekend_nights", 0), "stays_in_weekend_nights", min_val=0, max_val=60
        )
    except ValidationError as err:
        errors.update(err.errors)

    try:
        cleaned["stays_in_week_nights"] = _parse_int(
            raw_data.get("stays_in_week_nights", 0), "stays_in_week_nights", min_val=0, max_val=100
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 6. Guests (adults, children, babies)
    try:
        cleaned["adults"] = _parse_int(raw_data.get("adults", 1), "adults", min_val=0, max_val=20)
    except ValidationError as err:
        errors.update(err.errors)

    try:
        cleaned["children"] = _parse_int(raw_data.get("children", 0), "children", min_val=0, max_val=10)
    except ValidationError as err:
        errors.update(err.errors)

    try:
        cleaned["babies"] = _parse_int(raw_data.get("babies", 0), "babies", min_val=0, max_val=10)
    except ValidationError as err:
        errors.update(err.errors)

    # Cleaned dataset rule: at least 1 guest must be present
    if "adults" in cleaned and "children" in cleaned and "babies" in cleaned:
        if (cleaned["adults"] + cleaned["children"] + cleaned["babies"]) < 1:
            errors["guests"] = "At least one guest (adult, child, or baby) is required for a valid booking."

    # 7. Meal
    meal = str(raw_data.get("meal", "BB")).strip()
    if meal not in VALID_MEALS:
        errors["meal"] = f"Meal must be one of: {sorted(VALID_MEALS)}"
    else:
        cleaned["meal"] = meal

    # 8. Country
    country = str(raw_data.get("country", "PRT")).strip().upper()
    if not country or len(country) < 2 or len(country) > 3:
        errors["country"] = "Country must be a 2 or 3-letter ISO code (e.g. PRT, GBR, FRA, USA)."
    else:
        cleaned["country"] = country

    # 9. Market Segment & Distribution Channel
    market_segment = str(raw_data.get("market_segment", "Online TA")).strip()
    if market_segment not in VALID_MARKET_SEGMENTS:
        errors["market_segment"] = f"Market segment must be one of: {sorted(VALID_MARKET_SEGMENTS)}"
    else:
        cleaned["market_segment"] = market_segment

    distribution_channel = str(raw_data.get("distribution_channel", "TA/TO")).strip()
    if distribution_channel not in VALID_DISTRIBUTION_CHANNELS:
        errors["distribution_channel"] = f"Distribution channel must be one of: {sorted(VALID_DISTRIBUTION_CHANNELS)}"
    else:
        cleaned["distribution_channel"] = distribution_channel

    # 10. Repeated Guest
    repeated_raw = raw_data.get("is_repeated_guest", 0)
    if isinstance(repeated_raw, bool):
        cleaned["is_repeated_guest"] = 1 if repeated_raw else 0
    elif str(repeated_raw).strip().lower() in ("1", "yes", "true"):
        cleaned["is_repeated_guest"] = 1
    elif str(repeated_raw).strip().lower() in ("0", "no", "false", ""):
        cleaned["is_repeated_guest"] = 0
    else:
        errors["is_repeated_guest"] = "Repeated guest must be Yes/No or 1/0."

    # 11. History
    try:
        cleaned["previous_cancellations"] = _parse_int(
            raw_data.get("previous_cancellations", 0), "previous_cancellations", min_val=0, max_val=50
        )
    except ValidationError as err:
        errors.update(err.errors)

    try:
        cleaned["previous_bookings_not_canceled"] = _parse_int(
            raw_data.get("previous_bookings_not_canceled", 0), "previous_bookings_not_canceled", min_val=0, max_val=100
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 12. Room Types
    reserved_room = str(raw_data.get("reserved_room_type", "A")).strip().upper()
    if reserved_room not in VALID_RESERVED_ROOMS:
        errors["reserved_room_type"] = f"Reserved room type must be one of: {sorted(VALID_RESERVED_ROOMS)}"
    else:
        cleaned["reserved_room_type"] = reserved_room

    assigned_room = str(raw_data.get("assigned_room_type", reserved_room)).strip().upper()
    if assigned_room not in VALID_ASSIGNED_ROOMS:
        errors["assigned_room_type"] = f"Assigned room type must be one of: {sorted(VALID_ASSIGNED_ROOMS)}"
    else:
        cleaned["assigned_room_type"] = assigned_room

    # 13. Booking Changes
    try:
        cleaned["booking_changes"] = _parse_int(
            raw_data.get("booking_changes", 0), "booking_changes", min_val=0, max_val=50
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 14. Deposit Type
    deposit = str(raw_data.get("deposit_type", "No Deposit")).strip()
    if deposit not in VALID_DEPOSIT_TYPES:
        errors["deposit_type"] = f"Deposit type must be one of: {sorted(VALID_DEPOSIT_TYPES)}"
    else:
        cleaned["deposit_type"] = deposit

    # 15. Agent and Company (identifiers, optional)
    agent_val = raw_data.get("agent")
    if agent_val in (None, "", "None", "0", 0, "null"):
        cleaned["agent"] = None
    else:
        try:
            cleaned["agent"] = float(agent_val)
        except (ValueError, TypeError):
            cleaned["agent"] = str(agent_val)

    company_val = raw_data.get("company")
    if company_val in (None, "", "None", "0", 0, "null"):
        cleaned["company"] = None
    else:
        try:
            cleaned["company"] = float(company_val)
        except (ValueError, TypeError):
            cleaned["company"] = str(company_val)

    # 16. Days in Waiting List
    try:
        cleaned["days_in_waiting_list"] = _parse_int(
            raw_data.get("days_in_waiting_list", 0), "days_in_waiting_list", min_val=0, max_val=500
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 17. Customer Type
    cust_type = str(raw_data.get("customer_type", "Transient")).strip()
    if cust_type not in VALID_CUSTOMER_TYPES:
        errors["customer_type"] = f"Customer type must be one of: {sorted(VALID_CUSTOMER_TYPES)}"
    else:
        cleaned["customer_type"] = cust_type

    # 18. ADR (Average Daily Rate)
    try:
        cleaned["adr"] = _parse_float(
            raw_data.get("adr", 100.0), "adr", min_val=0.0, max_val=5000.0
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 19. Parking Spaces
    try:
        cleaned["required_car_parking_spaces"] = _parse_int(
            raw_data.get("required_car_parking_spaces", 0), "required_car_parking_spaces", min_val=0, max_val=10
        )
    except ValidationError as err:
        errors.update(err.errors)

    # 20. Special Requests
    try:
        cleaned["total_of_special_requests"] = _parse_int(
            raw_data.get("total_of_special_requests", 0), "total_of_special_requests", min_val=0, max_val=10
        )
    except ValidationError as err:
        errors.update(err.errors)

    if errors:
        first_msg = next(iter(errors.values()))
        raise ValidationError(first_msg, errors)

    return cleaned

"""
Feature Engineering and Leakage Handling Module for Hotel Booking Cancellation Prediction.

Member 3 Responsibilities:
1. Data leakage investigation and column removal
2. Domain-driven feature engineering
3. Feature validation and sanity checks
4. Feature selection / Feature & Target separation (X, y) for handoff to Member 4
"""

import pandas as pd
import numpy as np


def load_cleaned_data(filepath: str = "../../data/raw/hotel_bookings.csv") -> pd.DataFrame:
    """
    Loads raw data and reproduces Member 2's verified data cleaning pipeline:
    - Removes 31,994 exact duplicate rows
    - Removes 168 invalid records (0 guests, negative ADR, extreme ADR > 5000)

    Parameters
    ----------
    filepath : str
        Path to raw hotel_bookings.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame of shape (87228, 32).
    """
    df = pd.read_csv(filepath)
    df_dedup = df.drop_duplicates().copy()

    zero_guests_mask = (
        (df_dedup["adults"] == 0) &
        (df_dedup["children"].fillna(0) == 0) &
        (df_dedup["babies"] == 0)
    )
    neg_adr_mask = df_dedup["adr"] < 0
    extreme_adr_mask = df_dedup["adr"] > 5000

    clean_mask = (~zero_guests_mask) & (~neg_adr_mask) & (~extreme_adr_mask)
    df_clean = df_dedup[clean_mask].copy()

    return df_clean


def identify_leakage_columns() -> dict:
    """
    Returns a dictionary detailing the audit of potential data leakage columns.

    Returns
    -------
    dict
        Dictionary containing confirmed leakage columns and justification descriptions.
    """
    return {
        "confirmed_leakage": [
            "reservation_status",
            "reservation_status_date"
        ],
        "justifications": {
            "reservation_status": (
                "Direct post-outcome target leakage. Contains final booking states ('Check-Out', "
                "'Canceled', 'No-Show') which are 100% deterministically correlated with the target variable is_canceled."
            ),
            "reservation_status_date": (
                "Post-outcome target leakage. Represents the date when the reservation status was last updated "
                "(e.g., exact cancellation date or checkout date), which is unknown at booking time prior to cancellation."
            ),
            "days_in_waiting_list": (
                "Safe to keep. Represents waiting time before booking confirmation, which occurs prior to the stay "
                "outcome and is known at inference time."
            ),
            "booking_changes": (
                "Safe to keep. Represents amendments made by guests to their booking prior to arrival/cancellation."
            ),
            "deposit_type": (
                "Safe to keep. Represents deposit policy agreed upon at the time of reservation."
            )
        }
    }


def remove_leakage_columns(df: pd.DataFrame, leakage_cols: list = None) -> pd.DataFrame:
    """
    Safely removes confirmed target leakage columns from the dataframe.
    Ensures the target variable 'is_canceled' is never removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    leakage_cols : list, optional
        List of leakage column names to drop. Defaults to confirmed leakage list.

    Returns
    -------
    pd.DataFrame
        DataFrame with leakage columns removed.
    """
    if leakage_cols is None:
        leakage_cols = identify_leakage_columns()["confirmed_leakage"]

    # Safety check: Never drop target
    safe_drop_cols = [c for c in leakage_cols if c in df.columns and c != "is_canceled"]
    return df.drop(columns=safe_drop_cols).copy()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-relevant, leak-free features for cancellation prediction.

    Engineered Features:
    1. total_guests = adults + children + babies (handles missing children safely)
    2. total_stay = stays_in_week_nights + stays_in_weekend_nights
    3. is_family = 1 if (children + babies > 0) else 0
    4. room_changed = 1 if (reserved_room_type != assigned_room_type) else 0
    5. has_special_requests = 1 if (total_of_special_requests > 0) else 0
    6. has_previous_cancellations = 1 if (previous_cancellations > 0) else 0
    7. has_previous_bookings = 1 if (previous_bookings_not_canceled > 0) else 0
    8. is_weekend_only = 1 if (stays_in_weekend_nights > 0 and stays_in_week_nights == 0) else 0

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered features added.
    """
    df_feat = df.copy()

    # 1. Total Guests (safely handling 4 missing children values using fillna(0) for addition)
    children_safe = df_feat["children"].fillna(0)
    df_feat["total_guests"] = (df_feat["adults"] + children_safe + df_feat["babies"]).astype(int)

    # 2. Total Stay Duration (nights)
    df_feat["total_stay"] = (df_feat["stays_in_week_nights"] + df_feat["stays_in_weekend_nights"]).astype(int)

    # 3. Family Booking Indicator (binary)
    df_feat["is_family"] = ((children_safe + df_feat["babies"]) > 0).astype(int)

    # 4. Room Change / Discrepancy Indicator (binary)
    # Reflects operational room re-assignment, known before/at check-in
    df_feat["room_changed"] = (df_feat["reserved_room_type"] != df_feat["assigned_room_type"]).astype(int)

    # 5. Has Special Requests Indicator (binary)
    df_feat["has_special_requests"] = (df_feat["total_of_special_requests"] > 0).astype(int)

    # 6. Has Previous Cancellations History (binary)
    df_feat["has_previous_cancellations"] = (df_feat["previous_cancellations"] > 0).astype(int)

    # 7. Has Previous Successful Bookings (binary)
    df_feat["has_previous_bookings"] = (df_feat["previous_bookings_not_canceled"] > 0).astype(int)

    # 8. Weekend Only Stay (binary)
    df_feat["is_weekend_only"] = (
        (df_feat["stays_in_weekend_nights"] > 0) & (df_feat["stays_in_week_nights"] == 0)
    ).astype(int)

    return df_feat


def validate_features(df_before: pd.DataFrame, df_after: pd.DataFrame, new_feature_names: list) -> pd.DataFrame:
    """
    Validates engineered features for integrity, missingness, datatypes, and domain bounds.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before feature engineering.
    df_after : pd.DataFrame
        DataFrame after feature engineering.
    new_feature_names : list
        List of newly created feature names.

    Returns
    -------
    pd.DataFrame
        Validation summary table.
    """
    records = []
    for col in new_feature_names:
        series = df_after[col]
        records.append({
            "Feature Name": col,
            "Data Type": str(series.dtype),
            "Missing Count": int(series.isnull().sum()),
            "Min Value": series.min(),
            "Max Value": series.max(),
            "Unique Values Count": int(series.nunique()),
            "Sample Values": str(list(series.unique()[:5]))
        })
    return pd.DataFrame(records)


def prepare_feature_target(df: pd.DataFrame, target_col: str = "is_canceled") -> tuple:
    """
    Separates feature matrix X and target series y.
    Ensures target is separated cleanly without encoding or scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame containing features and target.
    target_col : str
        Target column name (default 'is_canceled').

    Returns
    -------
    tuple (pd.DataFrame, pd.Series)
        (X, y) feature matrix and target vector.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return X, y

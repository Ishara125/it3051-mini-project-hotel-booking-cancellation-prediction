"""
Data Quality Module for Hotel Booking Cancellation Prediction.

Member 2 Responsibilities:
- Missing-value analysis and assessment
- Duplicate detection and handling
- Invalid-value validation and domain constraint verification
- Statistical outlier detection (IQR) and domain review
"""

import pandas as pd
import numpy as np


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes missing values per column in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Summary table with Total Missing, Percentage Missing (%), and Data Type.
    """
    total_missing = df.isnull().sum()
    pct_missing = (total_missing / len(df)) * 100
    dtypes = df.dtypes

    missing_df = pd.DataFrame({
        "Total Missing": total_missing,
        "Percentage (%)": pct_missing.round(4),
        "Data Type": dtypes
    })

    return missing_df.sort_values(by="Total Missing", ascending=False)


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Checks for exact duplicate rows in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    dict
        Dictionary containing count, percentage, and boolean mask of duplicate rows.
    """
    duplicate_mask = df.duplicated()
    dup_count = int(duplicate_mask.sum())
    dup_pct = (dup_count / len(df)) * 100

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": round(dup_pct, 4),
        "duplicate_mask": duplicate_mask
    }


def check_invalid_values(df: pd.DataFrame) -> dict:
    """
    Validates logical, domain, and business constraints across dataset features.

    Checks include:
    - Zero guest bookings (adults == 0 and children == 0 and babies == 0)
    - Negative Average Daily Rate (ADR < 0)
    - Erroneous extreme ADR anomalies (e.g. ADR > 5000)
    - Zero stay duration bookings (weekend == 0 and week == 0)
    - Undefined placeholder categories in categorical fields

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    dict
        Dictionary of invalid masks and counts for each constraint.
    """
    zero_guests_mask = (
        (df["adults"] == 0) &
        (df["children"].fillna(0) == 0) &
        (df["babies"] == 0)
    )

    neg_adr_mask = df["adr"] < 0
    extreme_adr_mask = df["adr"] > 5000
    zero_stay_mask = (
        (df["stays_in_weekend_nights"] == 0) &
        (df["stays_in_week_nights"] == 0)
    )

    undefined_counts = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        cnt = int((df[col] == "Undefined").sum())
        if cnt > 0:
            undefined_counts[col] = cnt

    return {
        "zero_guests_count": int(zero_guests_mask.sum()),
        "zero_guests_mask": zero_guests_mask,
        "negative_adr_count": int(neg_adr_mask.sum()),
        "negative_adr_mask": neg_adr_mask,
        "extreme_adr_count": int(extreme_adr_mask.sum()),
        "extreme_adr_mask": extreme_adr_mask,
        "zero_stay_count": int(zero_stay_mask.sum()),
        "zero_stay_mask": zero_stay_mask,
        "undefined_categories": undefined_counts
    }


def detect_outliers_iqr(df: pd.DataFrame, numerical_cols: list = None) -> pd.DataFrame:
    """
    Performs statistical outlier detection using the Interquartile Range (IQR) method.

    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    numerical_cols : list, optional
        List of continuous/discrete numerical columns to evaluate.

    Returns
    -------
    pd.DataFrame
        Summary table containing Q1, Q3, IQR, bounds, outlier counts, and percentages.
    """
    if numerical_cols is None:
        exclude_cols = [
            "is_canceled", "is_repeated_guest",
            "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
            "agent", "company"
        ]
        numerical_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

    summary = []
    for col in numerical_cols:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        count = int(outlier_mask.sum())
        pct = (count / len(series)) * 100

        summary.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outlier Count": count,
            "Outlier (%)": round(pct, 2),
            "Min": round(float(series.min()), 2),
            "Max": round(float(series.max()), 2)
        })

    return pd.DataFrame(summary)

"""Reusable preprocessing utilities for hotel booking cancellation data.

Member 4 responsibilities are intentionally limited to feature/target preparation,
the stratified train/test split, and leakage-safe preprocessing. No estimator is
trained in this module.
"""

from __future__ import annotations

import gzip
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "is_canceled"
KNOWN_LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date")
IDENTIFIER_CATEGORICAL_COLUMNS = ("agent", "company")


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    leakage_columns: Iterable[str] = KNOWN_LEAKAGE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate predictors and target while defensively excluding known leakage.

    Member 3 normally removes the leakage columns before this function is called.
    The defensive removal prevents those fields from silently becoming predictors
    if an earlier notebook changes in the future.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataframe.")

    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    target_values = set(df[target_col].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_col}' must be binary (0/1); found {sorted(target_values)}."
        )

    present_leakage = [column for column in leakage_columns if column in df.columns]
    if present_leakage:
        warnings.warn(
            "Known post-outcome leakage columns were still present and have been "
            f"excluded from X: {present_leakage}. Review Member 3's leakage-removal step.",
            UserWarning,
            stacklevel=2,
        )

    excluded_columns = [target_col, *present_leakage]
    X = df.drop(columns=excluded_columns).copy()
    y = df[target_col].copy()

    if target_col in X.columns:
        raise AssertionError("The target must not be present in the predictor matrix.")

    return X, y, present_leakage


def identify_feature_types(
    X: pd.DataFrame,
    categorical_overrides: Iterable[str] = IDENTIFIER_CATEGORICAL_COLUMNS,
) -> tuple[list[str], list[str]]:
    """Identify numerical and categorical predictors from dtypes.

    ``agent`` and ``company`` are numeric-looking identifiers, not measured
    quantities. When present, they are therefore moved to the categorical list so
    their arbitrary code values are not given an artificial numerical ordering.
    """
    if TARGET_COLUMN in X.columns:
        raise ValueError(f"'{TARGET_COLUMN}' must be removed before type detection.")

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    for column in categorical_overrides:
        if column in X.columns:
            if column in numerical_features:
                numerical_features.remove(column)
            if column not in categorical_features:
                categorical_features.append(column)

    detected = set(numerical_features) | set(categorical_features)
    unsupported = [column for column in X.columns if column not in detected]
    if unsupported:
        dtype_details = {column: str(X[column].dtype) for column in unsupported}
        raise TypeError(f"Unsupported predictor dtypes detected: {dtype_details}")

    return numerical_features, categorical_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified split before fitting transformations."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if not X.index.equals(y.index):
        raise ValueError("X and y indexes must align before splitting.")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build numerical and categorical pipelines in one ColumnTransformer."""
    if not numerical_features and not categorical_features:
        raise ValueError("At least one predictor feature is required.")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Fit transformations on training data and only transform test data."""
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed


def target_distribution_table(
    y: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Return class counts and percentages for the full/train/test targets."""
    records = []
    for split_name, values in (("Full", y), ("Train", y_train), ("Test", y_test)):
        counts = values.value_counts().sort_index()
        percentages = values.value_counts(normalize=True).sort_index() * 100
        for class_label in (0, 1):
            records.append(
                {
                    "Dataset": split_name,
                    "Class": class_label,
                    "Meaning": "Not canceled" if class_label == 0 else "Canceled",
                    "Count": int(counts.get(class_label, 0)),
                    "Percentage (%)": round(float(percentages.get(class_label, 0.0)), 2),
                }
            )
    return pd.DataFrame(records)


def matrix_invalid_counts(matrix) -> dict[str, int]:
    """Count NaN and infinite values without densifying a sparse matrix."""
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return {
        "NaN count": int(np.isnan(values).sum()),
        "Infinity count": int(np.isinf(values).sum()),
    }


def build_verification_report(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_processed,
    X_test_processed,
    processed_feature_names: np.ndarray,
    target_col: str = TARGET_COLUMN,
    leakage_columns: Iterable[str] = KNOWN_LEAKAGE_COLUMNS,
    expected_test_size: float = 0.20,
) -> pd.DataFrame:
    """Create concise pass/fail checks for the completed preprocessing stage."""
    leakage_columns = tuple(leakage_columns)
    train_target_unchanged = y_train.equals(y.loc[y_train.index])
    test_target_unchanged = y_test.equals(y.loc[y_test.index])
    processed_names = [str(name) for name in processed_feature_names]
    leakage_in_processed_names = any(
        leakage_column in feature_name
        for leakage_column in leakage_columns
        for feature_name in processed_names
    )
    actual_test_size = len(X_test) / len(X)
    full_cancellation_rate = float(y.mean())
    max_stratification_difference = max(
        abs(float(y_train.mean()) - full_cancellation_rate),
        abs(float(y_test.mean()) - full_cancellation_rate),
    )
    train_invalid_counts = matrix_invalid_counts(X_train_processed)
    test_invalid_counts = matrix_invalid_counts(X_test_processed)

    checks = [
        ("Processed matrices created", X_train_processed is not None and X_test_processed is not None),
        ("Training feature/target row counts match", len(X_train) == len(y_train)),
        ("Test feature/target row counts match", len(X_test) == len(y_test)),
        ("Train/test processed column counts match", X_train_processed.shape[1] == X_test_processed.shape[1]),
        ("Processed feature-name count matches", len(processed_names) == X_train_processed.shape[1]),
        ("Training processed data has no NaN values", train_invalid_counts["NaN count"] == 0),
        ("Test processed data has no NaN values", test_invalid_counts["NaN count"] == 0),
        ("Training processed data has no infinite values", train_invalid_counts["Infinity count"] == 0),
        ("Test processed data has no infinite values", test_invalid_counts["Infinity count"] == 0),
        ("Training target remains binary", set(y_train.unique()).issubset({0, 1})),
        ("Test target remains binary", set(y_test.unique()).issubset({0, 1})),
        ("Targets remain unchanged", train_target_unchanged and test_target_unchanged),
        ("Target excluded from predictors", target_col not in X.columns),
        ("Known leakage excluded from predictors", not any(column in X.columns for column in leakage_columns)),
        ("Known leakage excluded from processed features", not leakage_in_processed_names),
        ("Train/test rows partition the full data", len(X_train) + len(X_test) == len(X)),
        ("Test proportion is 20% (within rounding)", abs(actual_test_size - expected_test_size) <= 1 / len(X)),
        ("Stratified class rates are preserved", max_stratification_difference < 0.001),
    ]

    return pd.DataFrame(
        {
            "Verification check": [name for name, _ in checks],
            "Status": ["PASS" if passed else "FAIL" for _, passed in checks],
        }
    )


def processed_preview(
    processed_matrix,
    processed_feature_names: np.ndarray,
    rows: int = 5,
    columns: int = 12,
) -> pd.DataFrame:
    """Create a small dense preview without densifying the complete matrix."""
    preview_matrix = processed_matrix[:rows, :columns]
    if sparse.issparse(preview_matrix):
        preview_matrix = preview_matrix.toarray()
    else:
        preview_matrix = np.asarray(preview_matrix)

    return pd.DataFrame(
        preview_matrix,
        columns=np.asarray(processed_feature_names)[:columns],
    )


def export_processed_csv(
    *,
    X_train_processed,
    X_test_processed,
    y_train: pd.Series,
    y_test: pd.Series,
    processed_feature_names: np.ndarray,
    output_path: str | Path,
    chunk_size: int = 2_000,
) -> Path:
    """Export train and test transformations to one CSV file.

    The matrices are written in small dense chunks so the complete sparse dataset
    is never converted to a large dense array. ``data_split`` preserves the
    train/test boundary, while ``source_index`` allows rows to be traced back to
    the cleaned dataframe. The target is appended unchanged and is not processed.
    """
    output_path = Path(output_path)
    is_compressed = output_path.suffixes[-2:] == [".csv", ".gz"]
    if output_path.suffix != ".csv" and not is_compressed:
        raise ValueError("output_path must end with '.csv' or '.csv.gz'.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if len(y_train) != X_train_processed.shape[0]:
        raise ValueError("Training matrix and target row counts do not match.")
    if len(y_test) != X_test_processed.shape[0]:
        raise ValueError("Test matrix and target row counts do not match.")
    if X_train_processed.shape[1] != len(processed_feature_names):
        raise ValueError("Processed feature-name count does not match the matrices.")
    if X_test_processed.shape[1] != len(processed_feature_names):
        raise ValueError("Train and test matrices must use the same processed features.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = np.asarray(processed_feature_names)
    first_chunk = True

    if is_compressed:
        output_file_context = gzip.open(
            output_path,
            mode="wt",
            encoding="utf-8",
            newline="",
        )
    else:
        output_file_context = output_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        )

    with output_file_context as output_file:
        for split_name, matrix, target in (
            ("train", X_train_processed, y_train),
            ("test", X_test_processed, y_test),
        ):
            for start in range(0, matrix.shape[0], chunk_size):
                stop = min(start + chunk_size, matrix.shape[0])
                matrix_chunk = matrix[start:stop]
                if sparse.issparse(matrix_chunk):
                    matrix_chunk = matrix_chunk.toarray()
                else:
                    matrix_chunk = np.asarray(matrix_chunk)

                chunk_df = pd.DataFrame(matrix_chunk, columns=feature_names)
                target_chunk = target.iloc[start:stop]
                chunk_df.insert(0, "source_index", target_chunk.index.to_numpy())
                chunk_df.insert(1, "data_split", split_name)
                chunk_df[TARGET_COLUMN] = target_chunk.to_numpy()
                chunk_df.to_csv(output_file, index=False, header=first_chunk)
                first_chunk = False

    return output_path

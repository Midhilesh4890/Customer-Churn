"""
Feature engineering module for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union

from src.utils.logger import get_logger
from src.config import SELECTED_FEATURES

logger = get_logger(__name__)


def create_tenure_groups(df: pd.DataFrame, column: str = "tenure") -> pd.DataFrame:
    """
    Create tenure groups from the tenure column.

    Args:
        df: Dataframe with the tenure column.
        column: Name of the tenure column.

    Returns:
        pandas.DataFrame: Dataframe with tenure group features.
    """
    logger.info("Creating tenure groups")

    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Define tenure groups
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = [
        "0-12 months",
        "13-24 months",
        "25-36 months",
        "37-48 months",
        "49-60 months",
        "61-72 months",
    ]

    # Create tenure group column
    result_df["tenure_group"] = pd.cut(
        result_df[column], bins=bins, labels=labels, right=True
    )

    # Convert to one-hot encoding
    tenure_dummies = pd.get_dummies(result_df["tenure_group"], prefix="tenure_group")
    result_df = pd.concat([result_df, tenure_dummies], axis=1)

    logger.info(f"Created {len(labels)} tenure group features")

    return result_df


def create_service_count(
    df: pd.DataFrame, service_columns: List[str] = None
) -> pd.DataFrame:
    """
    Create a feature that counts the number of services a customer has.

    Args:
        df: Dataframe with service columns.
        service_columns: List of service column names.

    Returns:
        pandas.DataFrame: Dataframe with service count feature.
    """
    logger.info("Creating service count feature")

    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    if service_columns is None:
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    # Filter to include only columns that exist in the dataframe
    service_columns = [col for col in service_columns if col in df.columns]

    # Create service count feature
    result_df["service_count"] = 0

    for col in service_columns:
        if pd.api.types.is_categorical_dtype(
            result_df[col]
        ) or pd.api.types.is_object_dtype(result_df[col]):
            # For categorical columns, count "Yes" values
            result_df["service_count"] += (result_df[col] == "Yes").astype(int)
        else:
            # For binary columns, assume 1 means "Yes"
            result_df["service_count"] += result_df[col]

    # Create service count groups
    result_df["service_count_group"] = pd.cut(
        result_df["service_count"],
        bins=[-1, 0, 2, 4, 6, 10],
        labels=["None", "Basic", "Medium", "High", "Complete"],
    )

    # Convert to one-hot encoding
    service_count_dummies = pd.get_dummies(
        result_df["service_count_group"], prefix="service_count"
    )
    result_df = pd.concat([result_df, service_count_dummies], axis=1)

    logger.info("Created service count feature")

    return result_df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between important variables.

    Args:
        df: Dataframe with features.

    Returns:
        pandas.DataFrame: Dataframe with interaction features.
    """
    logger.info("Creating interaction features")

    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Check if the relevant columns exist
    tenure_exists = "tenure" in result_df.columns
    monthly_charges_exists = "MonthlyCharges" in result_df.columns

    if tenure_exists and monthly_charges_exists:
        # Create tenure to MonthlyCharges ratio
        result_df["tenure_to_monthly_ratio"] = result_df["tenure"] / (
            result_df["MonthlyCharges"] + 1
        )
        logger.info("Created tenure to monthly charges ratio feature")

    # Create contract-tenure interaction
    contract_cols = [
        col for col in result_df.columns if "Contract" in col or "contract" in col
    ]

    if tenure_exists and contract_cols:
        for col in contract_cols:
            if pd.api.types.is_categorical_dtype(
                result_df[col]
            ) or pd.api.types.is_object_dtype(result_df[col]):
                # Skip categorical columns that haven't been one-hot encoded
                continue

            result_df[f"{col}_tenure"] = result_df[col] * result_df["tenure"]
            logger.info(f"Created {col}-tenure interaction feature")

    logger.info("Interaction feature creation complete")

    return result_df


def select_features(
    df: pd.DataFrame, selected_features: List[str] = None
) -> pd.DataFrame:
    """
    Select the specified features from the dataframe.

    Args:
        df: Dataframe with features.
        selected_features: List of feature names to select.

    Returns:
        pandas.DataFrame: Dataframe with selected features.
    """
    if selected_features is None:
        # If no features are specified, return the original dataframe
        return df

    # Filter to include only columns that exist in the dataframe
    valid_features = [col for col in selected_features if col in df.columns]

    if not valid_features:
        logger.warning(
            "No valid features found for selection, returning original dataframe"
        )
        return df

    logger.info(f"Selecting {len(valid_features)} features")

    return df[valid_features]


def engineer_features(
    df: pd.DataFrame,
    create_groups: bool = True,
    create_counts: bool = True,
    create_interactions: bool = True,
    select: bool = False,
    selected_features: List[str] = SELECTED_FEATURES,
) -> pd.DataFrame:
    """
    Apply feature engineering to the dataframe.

    Args:
        df: Dataframe to engineer features for.
        create_groups: Whether to create tenure groups.
        create_counts: Whether to create service count features.
        create_interactions: Whether to create interaction features.
        select: Whether to select only specific features.
        selected_features: List of feature names to select.

    Returns:
        pandas.DataFrame: Dataframe with engineered features.
    """
    logger.info("Applying feature engineering")

    result_df = df.copy()

    # Create tenure groups
    if create_groups:
        result_df = create_tenure_groups(result_df)

    # Create service count
    if create_counts:
        result_df = create_service_count(result_df)

    # Create interaction features
    if create_interactions:
        result_df = create_interaction_features(result_df)

    # Select features
    if select:
        result_df = select_features(result_df, selected_features)

    logger.info(f"Feature engineering completed. Shape: {result_df.shape}")

    return result_df

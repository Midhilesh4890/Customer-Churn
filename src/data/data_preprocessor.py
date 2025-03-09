import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    ID_COLUMN,
)

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data.

    Args:
        df: Raw dataframe to clean.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    logger.info("Cleaning raw data")

    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()

    # Convert TotalCharges to numeric, handling blank spaces
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"], errors="coerce"
    )

    # Fill missing TotalCharges values with 0 (for customers with tenure=0)
    null_mask = cleaned_df["TotalCharges"].isnull()
    cleaned_df.loc[null_mask, "TotalCharges"] = 0

    # Convert SeniorCitizen from 0/1 to No/Yes for consistency with other categorical features
    # also boolean will save more space than the integer
    cleaned_df["SeniorCitizen"] = cleaned_df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Convert Churn to binary (1/0)
    cleaned_df["Churn_Binary"] = (cleaned_df[TARGET] == "Yes").astype(int)

    logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")

    return cleaned_df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn_Binary",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        df: Dataframe to split.
        target: Target column name.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(
        f"Splitting data with test_size={test_size}, random_state={random_state}"
    )

    X = df.drop([target, TARGET, ID_COLUMN], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Data split completed: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    numerical_features: List[str] = NUMERICAL_FEATURES,
) -> ColumnTransformer:
    """
    Build a preprocessor for the data.

    Args:
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.

    Returns:
        sklearn.compose.ColumnTransformer: Preprocessor for the data.
    """
    logger.info("Building data preprocessor")

    # Preprocessing for numerical features
    # Pipeline is part of the Sklearn library
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Preprocessor built successfully")

    return preprocessor


def preprocess_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, preprocessor: ColumnTransformer = None
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Preprocess the data using the provided or newly created preprocessor.

    Args:
        X_train: Training features.
        X_test: Testing features.
        preprocessor: Optional preprocessor to use.

    Returns:
        Tuple: (X_train_processed, X_test_processed, preprocessor)
    """
    logger.info("Preprocessing data")

    if preprocessor is None:
        preprocessor = build_preprocessor()

    # Fit preprocessor on training data and transform both training and testing data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    feature_names = []

    # Get numerical feature names (these remain unchanged)
    numerical_features = preprocessor.transformers_[0][2]
    feature_names.extend(numerical_features)

    # Get categorical feature names after one-hot encoding
    categorical_features = preprocessor.transformers_[1][2]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    for i, category in enumerate(categorical_features):
        feature_names.extend([f"{category}_{val}" for val in ohe.categories_[i][1:]])

    # Convert to DataFrame for easier handling
    X_train_processed = pd.DataFrame(
        X_train_processed.toarray()
        if hasattr(X_train_processed, "toarray")
        else X_train_processed,
        columns=feature_names,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed.toarray()
        if hasattr(X_test_processed, "toarray")
        else X_test_processed,
        columns=feature_names,
    )

    logger.info(
        f"Data preprocessing completed: X_train={X_train_processed.shape}, "
        f"X_test={X_test_processed.shape}"
    )

    return X_train_processed, X_test_processed, preprocessor

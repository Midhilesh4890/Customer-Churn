"""
Data loader module for the churn prediction project.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union

from src.utils.logger import get_logger
from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR

logger = get_logger(__name__)


def load_raw_data(file_path: Union[str, Path] = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw data from the given file path.
    
    Args:
        file_path: Path to the raw data file.
        
    Returns:
        pandas.DataFrame: The loaded data.
    """
    logger.info(f"Loading raw data from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Union[str, Path] = PROCESSED_DATA_DIR
) -> Dict[str, Path]:
    """
    Save the processed data (training and testing sets) to the specified directory.
    
    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        output_dir: Directory to save the processed data files.
        
    Returns:
        Dict: Paths to the saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_dir}")

    files = {}

    try:
        # Save training data
        X_train_path = output_dir / "X_train.csv"
        X_train.to_csv(X_train_path, index=False)
        files["X_train"] = X_train_path

        # Save testing data
        X_test_path = output_dir / "X_test.csv"
        X_test.to_csv(X_test_path, index=False)
        files["X_test"] = X_test_path

        # Save training labels
        y_train_path = output_dir / "y_train.csv"
        y_train.to_csv(y_train_path, index=False)
        files["y_train"] = y_train_path

        # Save testing labels
        y_test_path = output_dir / "y_test.csv"
        y_test.to_csv(y_test_path, index=False)
        files["y_test"] = y_test_path

        logger.info("Successfully saved processed data")
        return files
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def load_processed_data(
    data_dir: Union[str, Path] = PROCESSED_DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load processed data from the specified directory.
    
    Args:
        data_dir: Directory containing the processed data files.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading processed data from {data_dir}")

    try:
        X_train = pd.read_csv(data_dir / "X_train.csv")
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
        y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

        logger.info(
            f"Successfully loaded processed data: "
            f"X_train={X_train.shape}, X_test={X_test.shape}, "
            f"y_train={y_train.shape}, y_test={y_test.shape}"
        )

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

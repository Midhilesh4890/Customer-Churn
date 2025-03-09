import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path

from src.pipeline.base import PipelineComponent
from src.data.data_loader import load_raw_data, save_processed_data, load_processed_data
from src.data.data_preprocessor import clean_data, split_data, preprocess_data
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
)


class DataLoaderComponent(PipelineComponent):
    """Component for loading raw data."""

    def __init__(self, data_path: Union[str, Path] = RAW_DATA_PATH):
        """
        Initialize the data loader component.

        Args:
            data_path: Path to the raw data file.
        """
        super().__init__(name="data_loader")
        self.data_path = data_path

    @PipelineComponent.log_execution_time
    def run(self, *args, **kwargs) -> pd.DataFrame:
        """
        Load the raw data from file.

        Returns:
            pandas.DataFrame: The loaded raw data.
        """
        self.logger.info(f"Loading data from {self.data_path}")
        return load_raw_data(self.data_path)


class DataCleanerComponent(PipelineComponent):
    """Component for cleaning raw data."""

    def __init__(self):
        """Initialize the data cleaner component."""
        super().__init__(name="data_cleaner")

    @PipelineComponent.log_execution_time
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data.

        Args:
            data: Raw dataframe to clean.

        Returns:
            pandas.DataFrame: Cleaned dataframe.
        """
        self.logger.info("Cleaning data")
        return clean_data(data)


class DataSplitterComponent(PipelineComponent):
    """Component for splitting data into training and testing sets."""

    def __init__(
        self,
        target: str = "Churn_Binary",
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """
        Initialize the data splitter component.

        Args:
            target: Target column name.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random seed for reproducibility.
        """
        super().__init__(name="data_splitter")
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    @PipelineComponent.log_execution_time
    def run(self, data: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split the data into training and testing sets.

        Args:
            data: Dataframe to split.

        Returns:
            Dict: Dictionary containing X_train, X_test, y_train, y_test.
        """
        self.logger.info(
            f"Splitting data with test_size={self.test_size}, "
            f"random_state={self.random_state}"
        )

        X_train, X_test, y_train, y_test = split_data(
            data, self.target, self.test_size, self.random_state
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_test_original": X_test.copy(),  # Keep a copy of the original test set
        }


class DataPreprocessorComponent(PipelineComponent):
    """Component for preprocessing data."""

    def __init__(
        self,
        categorical_features: List[str] = CATEGORICAL_FEATURES,
        numerical_features: List[str] = NUMERICAL_FEATURES,
        save_output: bool = True,
        output_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    ):
        """
        Initialize the data preprocessor component.

        Args:
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            save_output: Whether to save the processed data to disk.
            output_dir: Directory to save the processed data files.
        """
        super().__init__(name="data_preprocessor")
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.save_output = save_output
        self.output_dir = Path(output_dir)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
        """
        Preprocess the data.

        Args:
            data: Dictionary containing X_train, X_test, y_train, y_test.

        Returns:
            Dict: Dictionary containing processed data and preprocessor.
        """
        self.logger.info("Preprocessing data")

        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        X_train_processed, X_test_processed, preprocessor = preprocess_data(
            X_train, X_test
        )

        if self.save_output:
            self.logger.info(f"Saving processed data to {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_processed_data(
                X_train_processed, X_test_processed, y_train, y_test, self.output_dir
            )

        return {
            "X_train_processed": X_train_processed,
            "X_test_processed": X_test_processed,
            "y_train": y_train,
            "y_test": y_test,
            "X_test_original": data["X_test_original"],
            "preprocessor": preprocessor,
        }


class DataPipeline(PipelineComponent):
    """
    End-to-end data pipeline that combines loading, cleaning, splitting, and preprocessing.
    """

    def __init__(
        self,
        data_path: Union[str, Path] = RAW_DATA_PATH,
        target: str = "Churn_Binary",
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        save_output: bool = True,
        output_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    ):
        """
        Initialize the data pipeline.

        Args:
            data_path: Path to the raw data file.
            target: Target column name.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random seed for reproducibility.
            save_output: Whether to save the processed data to disk.
            output_dir: Directory to save the processed data files.
        """
        super().__init__(name="data_pipeline")

        # Create individual components
        self.loader = DataLoaderComponent(data_path)
        self.cleaner = DataCleanerComponent()
        self.splitter = DataSplitterComponent(target, test_size, random_state)
        self.preprocessor = DataPreprocessorComponent(
            save_output=save_output, output_dir=output_dir
        )

    @PipelineComponent.log_execution_time
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the complete data pipeline.

        Returns:
            Dict: Dictionary containing processed data and preprocessor.
        """
        # Run each component sequentially
        raw_data = self.loader.run()
        cleaned_data = self.cleaner.run(raw_data)
        split_data = self.splitter.run(cleaned_data)
        processed_data = self.preprocessor.run(split_data)

        return processed_data

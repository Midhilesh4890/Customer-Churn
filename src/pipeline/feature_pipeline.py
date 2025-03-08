"""
Feature engineering pipeline components for the churn prediction project.
"""
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path

from src.pipeline.base import PipelineComponent
from src.features.feature_engineering import (
    engineer_features,
    create_tenure_groups,
    create_service_count,
    create_interaction_features,
    select_features
)
from src.config import SELECTED_FEATURES


class FeatureEngineeringComponent(PipelineComponent):
    """
    Component for engineering features.
    """

    def __init__(
        self,
        create_groups: bool = True,
        create_counts: bool = True,
        create_interactions: bool = True,
        select: bool = False,
        selected_features: List[str] = SELECTED_FEATURES
    ):
        """
        Initialize the feature engineering component.
        
        Args:
            create_groups: Whether to create tenure groups.
            create_counts: Whether to create service count features.
            create_interactions: Whether to create interaction features.
            select: Whether to select only specific features.
            selected_features: List of feature names to select.
        """
        super().__init__(name="feature_engineering")
        self.create_groups = create_groups
        self.create_counts = create_counts
        self.create_interactions = create_interactions
        self.select = select
        self.selected_features = selected_features

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Engineer features for the data.
        
        Args:
            data: Dictionary containing data from previous pipeline stages.
            
        Returns:
            Dict: Dictionary with engineered features added.
        """
        self.logger.info("Engineering features")

        # Find X_train and X_test in the data dictionary
        # First check for direct keys
        X_train = data.get("X_train")
        X_test = data.get("X_test")

        # If not found, look inside the data_pipeline results
        if X_train is None and "data_pipeline" in data:
            pipeline_data = data["data_pipeline"]
            if isinstance(pipeline_data, dict):
                X_train = pipeline_data.get("X_train")
                X_test = pipeline_data.get("X_test")

        # As a fallback, check if we have data_splitter results
        if X_train is None and "data_splitter" in data:
            splitter_data = data["data_splitter"]
            if isinstance(splitter_data, dict):
                X_train = splitter_data.get("X_train")
                X_test = splitter_data.get("X_test")

        # Check if we found the required data
        if X_train is None or X_test is None:
            self.logger.error("X_train or X_test not found in input data")
            return data

        # Apply feature engineering to both sets
        X_train_engineered = engineer_features(
            X_train,
            create_groups=self.create_groups,
            create_counts=self.create_counts,
            create_interactions=self.create_interactions,
            select=self.select,
            selected_features=self.selected_features
        )

        X_test_engineered = engineer_features(
            X_test,
            create_groups=self.create_groups,
            create_counts=self.create_counts,
            create_interactions=self.create_interactions,
            select=self.select,
            selected_features=self.selected_features
        )

        # Create a new result dictionary with engineered features
        result = data.copy()
        result["X_train"] = X_train_engineered
        result["X_test"] = X_test_engineered

        return result


class SegmentMappingComponent(PipelineComponent):
    """
    Component for creating segment mappings between original categorical values 
    and processed feature columns.
    """

    def __init__(self, categorical_columns: List[str] = None):
        """
        Initialize the segment mapping component.
        
        Args:
            categorical_columns: List of categorical columns to create mappings for.
                                If None, will use ["Contract"] by default.
        """
        super().__init__(name="segment_mapping")
        self.categorical_columns = categorical_columns or ["Contract"]

    @staticmethod
    def create_segment_mapping(X_original, X_processed, original_column):
        """
        Create a mapping between original categorical values and processed feature columns.

        Args:
            X_original: Original features dataframe
            X_processed: Processed features dataframe
            original_column: Original categorical column name

        Returns:
            Dict: Mapping from original values to processed column names
        """
        mapping = {}
        unique_values = X_original[original_column].unique()

        # Find corresponding columns in processed data
        processed_columns = [
            col for col in X_processed.columns if col.startswith(f"{original_column}_")
        ]

        # Create mapping
        for value in unique_values:
            # Find which processed column corresponds to this value
            for col in processed_columns:
                # Extract the value from the column name
                col_value = col.replace(f"{original_column}_", "")
                if col_value.replace("_", " ") == value or col_value == value:
                    mapping[value] = col
                    break

        return mapping

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create segment mappings for the data.
        
        Args:
            data: Dictionary containing original and processed data.
            
        Returns:
            Dict: Dictionary with segment mappings added.
        """
        self.logger.info("Creating segment mappings")

        # Get the datasets - check multiple possible keys
        X_test_original = data.get("X_test_original")
        X_test_processed = data.get("X_test_processed")

        # If not found directly, check in the pipeline data
        if X_test_original is None and "data_pipeline" in data:
            pipeline_data = data["data_pipeline"]
            if isinstance(pipeline_data, dict):
                X_test_original = pipeline_data.get("X_test_original")

        # If still not found, try to use X_test as a fallback
        if X_test_original is None:
            X_test_original = data.get("X_test")
            self.logger.warning("Using X_test as fallback for X_test_original")

        # For X_test_processed, use X_test as fallback
        if X_test_processed is None:
            X_test_processed = data.get("X_test")
            self.logger.warning(
                "Using X_test as fallback for X_test_processed")

        if X_test_original is None or X_test_processed is None:
            self.logger.warning(
                "X_test_original or X_test_processed not found in data, skipping segment mapping"
            )
            return data

        # Create mappings for each categorical column
        segment_mappings = {}

        for column in self.categorical_columns:
            if column in X_test_original.columns:
                mapping = self.create_segment_mapping(
                    X_test_original, X_test_processed, column
                )
                segment_mappings[column] = mapping
                self.logger.info(f"Created mapping for {column}: {mapping}")
            else:
                self.logger.warning(
                    f"Column {column} not found in X_test_original")

        # Create a new result dictionary with segment mappings
        result = data.copy()
        result["segment_mappings"] = segment_mappings

        return result


class FeaturePipeline(PipelineComponent):
    """
    End-to-end feature pipeline that combines feature engineering and segment mapping.
    """

    def __init__(
        self,
        create_groups: bool = True,
        create_counts: bool = True,
        create_interactions: bool = True,
        select: bool = False,
        selected_features: List[str] = SELECTED_FEATURES,
        categorical_columns: List[str] = None
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            create_groups: Whether to create tenure groups.
            create_counts: Whether to create service count features.
            create_interactions: Whether to create interaction features.
            select: Whether to select only specific features.
            selected_features: List of feature names to select.
            categorical_columns: List of categorical columns to create mappings for.
        """
        super().__init__(name="feature_pipeline")

        # Create individual components
        self.feature_engineering = FeatureEngineeringComponent(
            create_groups=create_groups,
            create_counts=create_counts,
            create_interactions=create_interactions,
            select=select,
            selected_features=selected_features
        )

        self.segment_mapping = SegmentMappingComponent(
            categorical_columns=categorical_columns
        )

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete feature pipeline.
        
        Args:
            data: Dictionary containing data from the data pipeline.
            
        Returns:
            Dict: Dictionary with engineered features and segment mappings.
        """
        self.logger.info("Running feature pipeline")

        # Run each component sequentially
        engineered_data = self.feature_engineering.run(data)
        result = self.segment_mapping.run(engineered_data)

        return result

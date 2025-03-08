"""
Main pipeline for the churn prediction project.
"""
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

from src.pipeline.base import Pipeline
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.model_pipeline import ModelPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.pipeline.visualization_pipeline import VisualizationPipeline

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
    TEST_SIZE
)


def create_churn_pipeline(
    data_path: Union[str, Path] = RAW_DATA_PATH,
    models_to_train: List[str] = None,
    run_eda: bool = True,
    run_cv: bool = True,
    cv_folds: int = 5,
    run_threshold_opt: bool = True,
    save_models: bool = True,
    save_output: bool = True
) -> Pipeline:
    """
    Create a complete churn prediction pipeline.
    
    Args:
        data_path: Path to the raw data file.
        models_to_train: List of model types to train.
        run_eda: Whether to run EDA visualizations.
        run_cv: Whether to run cross-validation.
        cv_folds: Number of folds for cross-validation.
        run_threshold_opt: Whether to run threshold optimization.
        save_models: Whether to save trained models to disk.
        save_output: Whether to save processed data to disk.
        
    Returns:
        Pipeline: The complete churn prediction pipeline.
    """
    # Set default models to train
    if models_to_train is None:
        models_to_train = ["lr", "dt", "rf", "ensemble"]

    # Create the main pipeline
    pipeline = Pipeline(name="churn_prediction")

    # Create and add the data pipeline
    data_pipeline = DataPipeline(
        data_path=data_path,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        save_output=save_output,
        output_dir=PROCESSED_DATA_DIR
    )
    pipeline.add_component(data_pipeline)

    # Create and add the feature pipeline
    feature_pipeline = FeaturePipeline(
        create_groups=True,
        create_counts=True,
        create_interactions=True,
        categorical_columns=["Contract"]
    )
    pipeline.add_component(feature_pipeline)

    # Create and add the model pipeline
    model_pipeline = ModelPipeline(
        models_to_train=models_to_train,
        save_models=save_models,
        output_dir=MODELS_DIR,
        segment_column="Contract"
    )
    pipeline.add_component(model_pipeline)

    # Create and add the evaluation pipeline
    evaluation_pipeline = EvaluationPipeline(
        run_cv=run_cv,
        n_splits=cv_folds,
        run_threshold_opt=run_threshold_opt,
        output_dir=RESULTS_DIR,
        customer_id_column="customerID"
    )
    pipeline.add_component(evaluation_pipeline)

    # Create and add the visualization pipeline
    visualization_pipeline = VisualizationPipeline(
        run_eda=run_eda,
        output_dir=VISUALIZATIONS_DIR
    )
    pipeline.add_component(visualization_pipeline)

    return pipeline

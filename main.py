"""
Main module for the churn prediction project.
"""
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    SELECTED_FEATURES,
    RANDOM_STATE
)
from src.data.data_loader import load_raw_data, save_processed_data
from src.data.data_preprocessor import clean_data, split_data, preprocess_data
from src.features.feature_engineering import engineer_features
from src.models.model_factory import (
    create_model,
    train_model,
    evaluate_model,
    get_feature_importance,
    save_model
)
from src.evaluation.metrics import (
    calculate_metrics,
    evaluate_model_by_segment
)
from src.visualization.eda_visualizer import generate_eda_visualizations
from src.visualization.model_visualizer import generate_model_visualizations

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to raw data file"
    )

    parser.add_argument(
        "--skip_eda",
        action="store_true",
        help="Skip exploratory data analysis"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["lr", "dt", "rf", "ensemble", "all"],
        default=["all"],
        help="Models to train"
    )

    return parser.parse_args()


def main():
    """Run the churn prediction pipeline."""
    # Start timing
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    logger.info("Starting churn prediction pipeline")

    # Determine which models to train
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["lr", "dt", "rf", "ensemble"]

    # Load raw data
    df = load_raw_data(args.data_path)

    # Clean data
    df_cleaned = clean_data(df)

    # Generate EDA visualizations
    if not args.skip_eda:
        logger.info("Generating EDA visualizations")
        service_columns = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        _ = generate_eda_visualizations(
            df_cleaned,
            CATEGORICAL_FEATURES,
            NUMERICAL_FEATURES,
            service_columns,
            TARGET,
            VISUALIZATIONS_DIR
        )

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    # Engineer features
    X_train = engineer_features(
        X_train, create_groups=True, create_counts=True, create_interactions=True)
    X_test = engineer_features(
        X_test, create_groups=True, create_counts=True, create_interactions=True)

    # Preprocess data
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test)

    # Save processed data
    save_processed_data(X_train_processed, X_test_processed, y_train, y_test)

    # Train and evaluate models
    models = {}
    metrics = {}
    feature_importances = {}
    segment_metrics = {}

    for model_type in models_to_train:
        logger.info(f"Training {model_type} model")

        # Create and train model
        model = create_model(model_type)
        model = train_model(model, X_train_processed, y_train)

        # Save model
        save_model(model, f"{model_type}_model")

        # Evaluate model
        model_metrics = evaluate_model(model, X_test_processed, y_test)

        # Get feature importance
        feature_importance = get_feature_importance(
            model, X_train_processed.columns)

        # Evaluate by segment
        contract_segment_metrics = evaluate_model_by_segment(
            model,
            X_test,
            y_test,
            "Contract"
        )

        # Store results
        models[model_type] = model
        metrics[model_type] = model_metrics
        feature_importances[model_type] = feature_importance
        segment_metrics[model_type] = contract_segment_metrics

        logger.info(
            f"{model_type} model results: accuracy={model_metrics['accuracy']:.4f}, f1_score={model_metrics['f1_score']:.4f}")

    # Generate model visualizations
    _ = generate_model_visualizations(
        models,
        X_test_processed,
        y_test,
        metrics,
        feature_importances,
        segment_metrics,
        VISUALIZATIONS_DIR
    )

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(
        f"Pipeline execution completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()

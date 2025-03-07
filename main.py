"""
Main module for the churn prediction project.
"""
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
from sklearn.base import BaseEstimator

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
from src.models.cross_validation import (
    perform_kfold_cv,
    compare_cv_results,
    analyze_threshold_performance,
    get_optimal_threshold_by_metric
)
from src.evaluation.metrics import (
    calculate_metrics,
    evaluate_model_by_segment,
    get_classification_report
)
from src.visualization.eda_visualizer import generate_eda_visualizations
from src.visualization.advanced_model_visualizer import (
    plot_cv_results,
    plot_multiple_roc_curves,
    plot_multiple_pr_curves,
    plot_threshold_analysis,
    plot_optimal_thresholds,
    plot_confusion_matrices_at_thresholds
)

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

    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip cross-validation"
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--skip_threshold",
        action="store_true",
        help="Skip threshold analysis"
    )

    return parser.parse_args()


def save_test_predictions(
    models_dict: Dict[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    X_test_original: pd.DataFrame,
    customer_id_column: str = "customerID",
    output_dir: Path = Path("results")
) -> Path:
    """
    Save test predictions from multiple models to a CSV file.
    
    Args:
        models_dict: Dictionary of trained models.
        X_test: Processed test features.
        y_test: True test labels.
        X_test_original: Original unprocessed test features (for customer IDs).
        customer_id_column: Name of the customer ID column.
        output_dir: Directory to save the results.
        
    Returns:
        Path: Path to the saved CSV file.
    """
    logger.info("Saving test set predictions to CSV")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with customer IDs and true labels
    if customer_id_column in X_test_original.columns:
        results_df = pd.DataFrame({
            'CustomerID': X_test_original[customer_id_column],
            'True_Churn': y_test
        })
    else:
        # Create index-based customer IDs if customer ID column doesn't exist
        results_df = pd.DataFrame({
            'Index': range(len(y_test)),
            'True_Churn': y_test
        })

    # Add predictions from each model
    for model_name, model in models_dict.items():
        # Add binary predictions
        y_pred = model.predict(X_test)
        results_df[f'{model_name}_prediction'] = y_pred

        # Add probability predictions if available
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                results_df[f'{model_name}_probability'] = y_prob
            except Exception as e:
                logger.warning(
                    f"Could not get probabilities for {model_name}: {e}")

    # Save to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"test_predictions_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info(f"Test predictions saved to {csv_path}")

    return csv_path


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
        col for col in X_processed.columns if col.startswith(f"{original_column}_")]

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

    # Save original test data for reference (before preprocessing)
    X_test_original = X_test.copy()

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

    # Create mapping for segment analysis
    contract_mapping = create_segment_mapping(
        X_test_original, X_test_processed, "Contract")

    # Train and evaluate models
    models = {}
    metrics = {}
    feature_importances = {}
    segment_metrics = {}
    y_prob_dict = {}  # Store probabilities for ROC curve comparison

    for model_type in models_to_train:
        logger.info(f"Training {model_type} model")

        # Create and train model
        model = create_model(model_type)
        model = train_model(model, X_train_processed, y_train)

        # Save model
        save_model(model, f"{model_type}_model")

        # Evaluate model
        model_metrics = evaluate_model(model, X_test_processed, y_test)

        # Store probabilities if model supports it
        if hasattr(model, 'predict_proba'):
            try:
                y_prob_dict[model_type] = model.predict_proba(X_test_processed)[
                    :, 1]
            except Exception as e:
                logger.warning(
                    f"Could not get probabilities for {model_type}: {e}")

        # Get feature importance
        feature_importance = get_feature_importance(
            model, X_train_processed.columns)

        # Evaluate by contract type segment (using processed features and column names)
        # We need to evaluate each contract type separately using the processed feature columns
        contract_segment_metrics = {}

        for contract_type, processed_col in contract_mapping.items():
            # Create a segment mask for this contract type
            segment_indices = X_test_processed[X_test_processed[processed_col] == 1].index

            # Skip if no samples in this segment
            if len(segment_indices) == 0:
                continue

            # Get segment data
            X_segment = X_test_processed.loc[segment_indices]
            y_segment = y_test.iloc[segment_indices] if isinstance(
                y_test, pd.Series) else y_test[segment_indices]

            # Make predictions
            y_pred = model.predict(X_segment)

            # Calculate probabilities if model supports it
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_segment)[:, 1]
                    metrics = calculate_metrics(y_segment, y_pred, y_prob)
                except Exception as e:
                    logger.warning(
                        f"Could not get probabilities for segment {contract_type}: {e}")
                    metrics = calculate_metrics(y_segment, y_pred)
            else:
                metrics = calculate_metrics(y_segment, y_pred)

            # Add churn rate
            metrics['churn_rate'] = y_segment.mean()
            metrics['count'] = len(y_segment)

            # Store metrics for this segment
            contract_segment_metrics[contract_type] = metrics

            logger.info(
                f"Segment {contract_type}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, churn_rate={metrics['churn_rate']:.4f}, count={metrics['count']}")

        # Store results
        models[model_type] = model
        metrics[model_type] = model_metrics
        feature_importances[model_type] = feature_importance
        segment_metrics[model_type] = contract_segment_metrics

        logger.info(
            f"{model_type} model results: accuracy={model_metrics['accuracy']:.4f}, f1_score={model_metrics['f1_score']:.4f}")

    # Save test set predictions to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    test_predictions_path = save_test_predictions(
        models,
        X_test_processed,
        y_test,
        X_test_original,
        customer_id_column="customerID",
        output_dir=results_dir
    )
    logger.info(f"Test predictions saved to {test_predictions_path}")

    # Create advanced visualizations directory
    advanced_viz_dir = Path(VISUALIZATIONS_DIR) / "advanced"
    advanced_viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate advanced visualizations if we have probability predictions
    if y_prob_dict:
        logger.info("Generating advanced model visualizations")

        # Plot multiple ROC curves
        _ = plot_multiple_roc_curves(
            y_test,
            y_prob_dict,
            output_path=advanced_viz_dir / "roc_curves_comparison.png"
        )

        # Plot multiple PR curves
        _ = plot_multiple_pr_curves(
            y_test,
            y_prob_dict,
            output_path=advanced_viz_dir / "pr_curves_comparison.png"
        )

    # Perform cross-validation unless explicitly skipped
    if not args.skip_cv:
        logger.info(f"Performing {args.cv_folds}-fold cross-validation")

        # We'll use unprocessed data and handle preprocessing in the cross-validation
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test]) if isinstance(
            y_train, pd.Series) else np.concatenate([y_train, y_test])

        try:
            # Perform cross-validation
            cv_results = perform_kfold_cv(
                models,
                X_combined,
                y_combined,
                n_splits=args.cv_folds
            )

            # Compare cross-validation results
            cv_comparison_df = compare_cv_results(cv_results)

            # Plot cross-validation results
            _ = plot_cv_results(
                cv_comparison_df,
                output_path=advanced_viz_dir / "cross_validation_results.png"
            )
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")

    # Perform threshold analysis unless explicitly skipped
    if not args.skip_threshold and y_prob_dict:
        logger.info("Performing threshold analysis")

        # Dictionary to store models with optimal thresholds
        optimal_models = {}

        for model_name, y_prob in y_prob_dict.items():
            logger.info(f"Analyzing thresholds for {model_name} model")

            # Create model-specific directory
            model_threshold_dir = advanced_viz_dir / model_name
            model_threshold_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Analyze thresholds
                threshold_results = analyze_threshold_performance(
                    y_test, y_prob)

                # Plot threshold analysis
                _ = plot_threshold_analysis(
                    threshold_results,
                    output_path=model_threshold_dir / "threshold_analysis.png"
                )

                # Plot confusion matrices at different thresholds
                _ = plot_confusion_matrices_at_thresholds(
                    y_test,
                    y_prob,
                    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                    output_path=model_threshold_dir / "confusion_matrices_at_thresholds.png"
                )

                # Get optimal threshold for different metrics
                metrics_to_optimize = [
                    'f1', 'precision', 'recall', 'balanced_accuracy']

                # For F1 score, create a model wrapper with optimal threshold
                optimal_threshold, optimal_score = get_optimal_threshold_by_metric(
                    y_test, y_prob, 'f1'
                )
                logger.info(
                    f"{model_name} optimal threshold for f1: {optimal_threshold:.4f} (score: {optimal_score:.4f})")

                # Create a class that wraps the model with optimal threshold
                class OptimalThresholdWrapper:
                    def __init__(self, base_model, threshold):
                        self.base_model = base_model
                        self.threshold = threshold

                    def predict(self, X):
                        probas = self.base_model.predict_proba(X)[:, 1]
                        return (probas >= self.threshold).astype(int)

                    def predict_proba(self, X):
                        return self.base_model.predict_proba(X)

                # Add wrapped model to dictionary
                optimal_models[f"{model_name}_optimal"] = OptimalThresholdWrapper(
                    models[model_name], optimal_threshold)

                # Log other metrics' optimal thresholds
                # Skip f1 as we already processed it
                for metric in metrics_to_optimize[1:]:
                    optimal_threshold, optimal_score = get_optimal_threshold_by_metric(
                        y_test, y_prob, metric
                    )
                    logger.info(
                        f"{model_name} optimal threshold for {metric}: {optimal_threshold:.4f} (score: {optimal_score:.4f})")

            except Exception as e:
                logger.error(
                    f"Error during threshold analysis for {model_name}: {e}")

        try:
            # Plot optimal thresholds comparison
            _ = plot_optimal_thresholds(
                y_test,
                y_prob_dict,
                output_path=advanced_viz_dir / "optimal_thresholds_comparison.png"
            )

            # Save predictions with optimal thresholds
            if optimal_models:
                optimal_predictions_path = save_test_predictions(
                    optimal_models,
                    X_test_processed,
                    y_test,
                    X_test_original,
                    customer_id_column="customerID",
                    output_dir=results_dir
                )
                logger.info(
                    f"Optimal threshold predictions saved to {optimal_predictions_path}")

        except Exception as e:
            logger.error(f"Error in threshold optimization: {e}")

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(
        f"Pipeline execution completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()

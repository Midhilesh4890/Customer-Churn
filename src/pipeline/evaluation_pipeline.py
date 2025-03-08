"""
Evaluation pipeline components for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
from sklearn.base import BaseEstimator

from src.pipeline.base import PipelineComponent
from src.models.cross_validation import (
    perform_kfold_cv,
    compare_cv_results,
    analyze_threshold_performance,
    get_optimal_threshold_by_metric,
)
from src.evaluation.metrics import (
    calculate_metrics,
    get_classification_report
)
from src.config import RANDOM_STATE


class CrossValidationComponent(PipelineComponent):
    """
    Component for performing cross-validation on models.
    """

    def __init__(self, n_splits: int = 5, random_state: int = RANDOM_STATE):
        """
        Initialize the cross-validation component.
        
        Args:
            n_splits: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.
        """
        super().__init__(name="cross_validation")
        self.n_splits = n_splits
        self.random_state = random_state

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation on models.
        
        Args:
            data: Dictionary containing trained models and data.
            
        Returns:
            Dict: Dictionary with cross-validation results added.
        """
        self.logger.info(f"Performing {self.n_splits}-fold cross-validation")

        # Get the required data
        models = data.get("models", {})
        X_train = data.get("X_train")
        X_test = data.get("X_test")
        y_train = data.get("y_train")
        y_test = data.get("y_test")

        if not models or X_train is None or y_train is None:
            self.logger.warning("Required data not found for cross-validation")
            return data

        try:
            # Combine train and test data for cross-validation
            X_combined = pd.concat([X_train, X_test])
            y_combined = (
                pd.concat([y_train, y_test])
                if isinstance(y_train, pd.Series)
                else np.concatenate([y_train, y_test])
            )

            # Perform cross-validation
            cv_results = perform_kfold_cv(
                models, X_combined, y_combined, n_splits=self.n_splits
            )

            # Compare cross-validation results
            cv_comparison_df = compare_cv_results(cv_results)

            # Create a new result dictionary with cross-validation results
            result = data.copy()
            result["cv_results"] = cv_results
            result["cv_comparison_df"] = cv_comparison_df

            return result

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {e}")
            return data


class ThresholdOptimizerComponent(PipelineComponent):
    """
    Component for optimizing probability thresholds for classification.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize the threshold optimizer component.
        
        Args:
            metrics: List of metrics to optimize thresholds for.
        """
        super().__init__(name="threshold_optimizer")
        self.metrics = metrics or [
            "f1", "precision", "recall", "balanced_accuracy"]

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize probability thresholds for models.
        
        Args:
            data: Dictionary containing trained models and data.
            
        Returns:
            Dict: Dictionary with threshold optimization results added.
        """
        self.logger.info("Optimizing probability thresholds")

        # Get the required data
        y_prob_dict = data.get("y_prob_dict", {})
        y_test = data.get("y_test")
        models = data.get("models", {})
        X_test_processed = data.get("X_test_processed")
        X_test_original = data.get("X_test_original")

        if not y_prob_dict or y_test is None:
            self.logger.warning(
                "Required data not found for threshold optimization")
            return data

        # Dictionary to store threshold results
        threshold_results = {}

        # Dictionary to store models with optimal thresholds
        optimal_models = {}

        # Dictionary to store optimal thresholds
        optimal_thresholds = {}

        # For each model with probability predictions
        for model_name, y_prob in y_prob_dict.items():
            self.logger.info(f"Analyzing thresholds for {model_name} model")

            try:
                # Analyze thresholds
                model_threshold_results = analyze_threshold_performance(
                    y_test, y_prob)
                threshold_results[model_name] = model_threshold_results

                # Store optimal thresholds for each metric
                model_optimal_thresholds = {}

                for metric in self.metrics:
                    optimal_threshold, optimal_score = get_optimal_threshold_by_metric(
                        y_test, y_prob, metric
                    )

                    model_optimal_thresholds[metric] = {
                        "threshold": optimal_threshold,
                        "score": optimal_score
                    }

                    self.logger.info(
                        f"{model_name} optimal threshold for {metric}: "
                        f"{optimal_threshold:.4f} (score: {optimal_score:.4f})"
                    )

                # Store optimal thresholds for this model
                optimal_thresholds[model_name] = model_optimal_thresholds

                # Create a class that wraps the model with optimal threshold for F1
                class OptimalThresholdWrapper:
                    def __init__(self, base_model, threshold):
                        self.base_model = base_model
                        self.threshold = threshold

                    def predict(self, X):
                        probas = self.base_model.predict_proba(X)[:, 1]
                        return (probas >= self.threshold).astype(int)

                    def predict_proba(self, X):
                        return self.base_model.predict_proba(X)

                # Add wrapped model with optimal F1 threshold to dictionary
                f1_threshold = model_optimal_thresholds["f1"]["threshold"]
                optimal_models[f"{model_name}_optimal"] = OptimalThresholdWrapper(
                    models[model_name], f1_threshold
                )

            except Exception as e:
                self.logger.error(
                    f"Error during threshold analysis for {model_name}: {e}")

        # Create a new result dictionary with threshold optimization results
        result = data.copy()
        result["threshold_results"] = threshold_results
        result["optimal_thresholds"] = optimal_thresholds
        result["optimal_models"] = optimal_models

        return result


class TestPredictionsComponent(PipelineComponent):
    """
    Component for generating and saving test set predictions.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = Path("results"),
        customer_id_column: str = "customerID"
    ):
        """
        Initialize the test predictions component.
        
        Args:
            output_dir: Directory to save the prediction results.
            customer_id_column: Name of the customer ID column.
        """
        super().__init__(name="test_predictions")
        self.output_dir = Path(output_dir)
        self.customer_id_column = customer_id_column

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and save test set predictions.
        
        Args:
            data: Dictionary containing trained models and data.
            
        Returns:
            Dict: Dictionary with test prediction results added.
        """
        self.logger.info("Generating test set predictions")

        # Get the required data
        models_dict = data.get("models", {})
        optimal_models = data.get("optimal_models", {})
        X_test_processed = data.get("X_test_processed")
        y_test = data.get("y_test")
        X_test_original = data.get("X_test_original")

        if not models_dict or X_test_processed is None or y_test is None:
            self.logger.warning("Required data not found for test predictions")
            return data

        # Combine regular and optimal models
        all_models = {**models_dict, **optimal_models}

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame with customer IDs and true labels
        if self.customer_id_column in X_test_original.columns:
            results_df = pd.DataFrame({
                "CustomerID": X_test_original[self.customer_id_column],
                "True_Churn": y_test
            })
        else:
            # Create index-based customer IDs if customer ID column doesn't exist
            results_df = pd.DataFrame({
                "Index": range(len(y_test)),
                "True_Churn": y_test
            })

        # Add predictions from each model
        for model_name, model in all_models.items():
            # Add binary predictions
            y_pred = model.predict(X_test_processed)
            results_df[f"{model_name}_prediction"] = y_pred

            # Add probability predictions if available
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_processed)[:, 1]
                    results_df[f"{model_name}_probability"] = y_prob
                except Exception as e:
                    self.logger.warning(
                        f"Could not get probabilities for {model_name}: {e}"
                    )

        # Save to CSV
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"test_predictions_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)

        self.logger.info(f"Test predictions saved to {csv_path}")

        # Create a new result dictionary with test prediction results
        result = data.copy()
        result["test_predictions_path"] = csv_path
        result["test_predictions_df"] = results_df

        return result


class EvaluationPipeline(PipelineComponent):
    """
    End-to-end evaluation pipeline that combines cross-validation, 
    threshold optimization, and test predictions.
    """

    def __init__(
        self,
        run_cv: bool = True,
        n_splits: int = 5,
        run_threshold_opt: bool = True,
        metrics: List[str] = None,
        output_dir: Union[str, Path] = Path("results"),
        customer_id_column: str = "customerID"
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            run_cv: Whether to run cross-validation.
            n_splits: Number of folds for cross-validation.
            run_threshold_opt: Whether to run threshold optimization.
            metrics: List of metrics to optimize thresholds for.
            output_dir: Directory to save the results.
            customer_id_column: Name of the customer ID column.
        """
        super().__init__(name="evaluation_pipeline")

        # Create individual components
        self.components = []

        if run_cv:
            self.cv = CrossValidationComponent(n_splits=n_splits)
            self.components.append(self.cv)

        if run_threshold_opt:
            self.threshold_opt = ThresholdOptimizerComponent(metrics=metrics)
            self.components.append(self.threshold_opt)

        self.test_predictions = TestPredictionsComponent(
            output_dir=output_dir,
            customer_id_column=customer_id_column
        )
        self.components.append(self.test_predictions)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            data: Dictionary containing data from previous pipelines.
            
        Returns:
            Dict: Dictionary with evaluation results.
        """
        result = data.copy()

        # Run each component sequentially
        for component in self.components:
            result = component.run(result)

        return result

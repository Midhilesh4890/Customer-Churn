"""
Model pipeline components for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
from sklearn.base import BaseEstimator

from src.pipeline.base import PipelineComponent
from src.models.model_factory import (
    create_model,
    train_model,
    evaluate_model,
    get_feature_importance,
    save_model,
)
from src.config import MODELS_DIR


class ModelTrainerComponent(PipelineComponent):
    """
    Component for creating and training models.
    """

    def __init__(
        self,
        models_to_train: List[str] = None,
        save_models: bool = True,
        output_dir: Union[str, Path] = MODELS_DIR,
    ):
        """
        Initialize the model trainer component.

        Args:
            models_to_train: List of model types to train ('lr', 'dt', 'rf', 'ensemble').
                            If None, will train all models.
            save_models: Whether to save trained models to disk.
            output_dir: Directory to save the model files.
        """
        super().__init__(name="model_trainer")
        self.models_to_train = models_to_train or ["lr", "dt", "rf", "ensemble"]
        self.save_models = save_models
        self.output_dir = Path(output_dir)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and train models.

        Args:
            data: Dictionary containing processed data.

        Returns:
            Dict: Dictionary with trained models, metrics, etc.
        """
        self.logger.info(f"Training models: {self.models_to_train}")

        # Get the processed data
        X_train_processed = data.get("X_train_processed")
        X_test_processed = data.get("X_test_processed")
        y_train = data.get("y_train")
        y_test = data.get("y_test")

        if (
            X_train_processed is None
            or X_test_processed is None
            or y_train is None
            or y_test is None
        ):
            self.logger.error("Required data not found in input dictionary")
            return data

        # Create dictionaries to store results
        models = {}
        metrics = {}
        feature_importances = {}
        y_prob_dict = {}

        # Train each model
        for model_type in self.models_to_train:
            self.logger.info(f"Training {model_type} model")

            # Create and train model
            model = create_model(model_type)
            model = train_model(model, X_train_processed, y_train)

            # Save model if requested
            if self.save_models:
                save_model(model, f"{model_type}_model", self.output_dir)

            # Evaluate model
            model_metrics = evaluate_model(model, X_test_processed, y_test)

            # Store probabilities if model supports it
            if hasattr(model, "predict_proba"):
                try:
                    y_prob_dict[model_type] = model.predict_proba(X_test_processed)[
                        :, 1
                    ]
                except Exception as e:
                    self.logger.warning(
                        f"Could not get probabilities for {model_type}: {e}"
                    )

            # Get feature importance
            feature_importance = get_feature_importance(
                model, X_train_processed.columns
            )

            # Store results
            models[model_type] = model
            metrics[model_type] = model_metrics
            feature_importances[model_type] = feature_importance

            self.logger.info(
                f"{model_type} model results: "
                f"accuracy={model_metrics['accuracy']:.4f}, "
                f"f1_score={model_metrics['f1_score']:.4f}"
            )

        # Create a new result dictionary with trained models
        result = data.copy()
        result["models"] = models
        result["metrics"] = metrics
        result["feature_importances"] = feature_importances
        result["y_prob_dict"] = y_prob_dict

        return result


class SegmentEvaluationComponent(PipelineComponent):
    """
    Component for evaluating models by segment.
    """

    def __init__(self, segment_column: str = "Contract"):
        """
        Initialize the segment evaluation component.

        Args:
            segment_column: Column to segment by.
        """
        super().__init__(name="segment_evaluation")
        self.segment_column = segment_column

    @staticmethod
    def evaluate_segment(
        model: BaseEstimator,
        X_segment: pd.DataFrame,
        y_segment: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate a model on a specific segment.

        Args:
            model: Trained model to evaluate.
            X_segment: Features for the segment.
            y_segment: Labels for the segment.

        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        from src.evaluation.metrics import calculate_metrics

        # Make predictions
        y_pred = model.predict(X_segment)

        # Calculate probabilities if model supports it
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_segment)[:, 1]
                metrics = calculate_metrics(y_segment, y_pred, y_prob)
            except Exception as e:
                metrics = calculate_metrics(y_segment, y_pred)
        else:
            metrics = calculate_metrics(y_segment, y_pred)

        # Add churn rate
        metrics["churn_rate"] = y_segment.mean()
        metrics["count"] = len(y_segment)

        return metrics

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate models by segment.

        Args:
            data: Dictionary containing trained models and processed data.

        Returns:
            Dict: Dictionary with segment evaluation results added.
        """
        self.logger.info(f"Evaluating models by segment: {self.segment_column}")

        # Get the required data
        models = data.get("models", {})
        X_test_processed = data.get("X_test_processed")
        y_test = data.get("y_test")
        X_test_original = data.get("X_test_original")
        segment_mappings = data.get("segment_mappings", {})

        if not models or X_test_processed is None or y_test is None:
            self.logger.error("Required data not found in input dictionary")
            return data

        # Get mapping for the segment column
        contract_mapping = segment_mappings.get(self.segment_column, {})

        if not contract_mapping:
            self.logger.warning(
                f"No mapping found for segment column: {self.segment_column}"
            )
            return data

        # Dictionary to store segment evaluation results
        segment_metrics = {}

        # Evaluate each model by segment
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name} model by segment")

            # Dictionary to store metrics for each segment
            contract_segment_metrics = {}

            # Evaluate for each segment
            for contract_type, processed_col in contract_mapping.items():
                # Create a segment mask for this contract type
                segment_indices = X_test_processed[
                    X_test_processed[processed_col] == 1
                ].index

                # Skip if no samples in this segment
                if len(segment_indices) == 0:
                    continue

                # Get segment data
                X_segment = X_test_processed.loc[segment_indices]
                y_segment = (
                    y_test.iloc[segment_indices]
                    if isinstance(y_test, pd.Series)
                    else y_test[segment_indices]
                )

                # Evaluate the segment
                metrics = self.evaluate_segment(model, X_segment, y_segment)

                # Store metrics for this segment
                contract_segment_metrics[contract_type] = metrics

                self.logger.info(
                    f"Segment {contract_type}: "
                    f"accuracy={metrics['accuracy']:.4f}, "
                    f"f1={metrics['f1']:.4f}, "
                    f"churn_rate={metrics['churn_rate']:.4f}, "
                    f"count={metrics['count']}"
                )

            # Store results for this model
            segment_metrics[model_name] = contract_segment_metrics

        # Create a new result dictionary with segment evaluation results
        result = data.copy()
        result["segment_metrics"] = segment_metrics

        return result


class ModelPipeline(PipelineComponent):
    """
    End-to-end model pipeline that combines model training and segment evaluation.
    """

    def __init__(
        self,
        models_to_train: List[str] = None,
        save_models: bool = True,
        output_dir: Union[str, Path] = MODELS_DIR,
        segment_column: str = "Contract",
    ):
        """
        Initialize the model pipeline.

        Args:
            models_to_train: List of model types to train.
            save_models: Whether to save trained models to disk.
            output_dir: Directory to save the model files.
            segment_column: Column to segment by for evaluation.
        """
        super().__init__(name="model_pipeline")

        # Create individual components
        self.model_trainer = ModelTrainerComponent(
            models_to_train=models_to_train,
            save_models=save_models,
            output_dir=output_dir,
        )

        self.segment_evaluation = SegmentEvaluationComponent(
            segment_column=segment_column
        )

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete model pipeline.

        Args:
            data: Dictionary containing data from previous pipelines.

        Returns:
            Dict: Dictionary with trained models and evaluation results.
        """
        # Run each component sequentially
        trained_data = self.model_trainer.run(data)
        result = self.segment_evaluation.run(trained_data)

        return result

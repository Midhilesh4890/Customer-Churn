import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Union[List, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (optional).

    Returns:
        Dict: Dictionary of evaluation metrics.
    """
    logger.info("Calculating evaluation metrics")

    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # Add AUC if probabilities are provided
    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Add confusion matrix metrics
    metrics.update(
        {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        }
    )

    # Calculate additional metrics
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["balanced_accuracy"] = (metrics["specificity"] + metrics["sensitivity"]) / 2

    logger.info(
        f"Metrics calculated: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
    )

    return metrics


def get_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    target_names: List[str] = ["Not Churn", "Churn"],
) -> Dict[str, Any]:
    """
    Get a detailed classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Names of target classes.

    Returns:
        Dict: Classification report as a dictionary.
    """
    logger.info("Generating classification report")

    # Generate classification report as a dictionary
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    logger.info("Classification report generated")

    return report


def get_roc_curve_data(
    y_true: Union[List, np.ndarray], y_prob: Union[List, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Tuple: (fpr, tpr, thresholds)
    """
    logger.info("Calculating ROC curve data")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return fpr, tpr, thresholds


def get_precision_recall_curve_data(
    y_true: Union[List, np.ndarray], y_prob: Union[List, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get precision-recall curve data.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Tuple: (precision, recall, thresholds)
    """
    logger.info("Calculating precision-recall curve data")

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    return precision, recall, thresholds


def evaluate_model_by_segment(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    segment_column: str,
    segment_values: Dict[Any, str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance by segment.

    Args:
        model: Trained model.
        X: Features.
        y: True labels.
        segment_column: Column to segment by.
        segment_values: Mapping of segment values to names (optional).

    Returns:
        Dict: Dictionary of metrics by segment.
    """
    logger.info(f"Evaluating model by segment: {segment_column}")

    if segment_column not in X.columns:
        logger.error(f"Segment column '{segment_column}' not found in features")
        return {}

    # Get unique segment values
    unique_segments = X[segment_column].unique()

    # Create dictionary to store results
    segment_metrics = {}

    # Evaluate for each segment
    for segment in unique_segments:
        # Get indices for this segment
        segment_mask = X[segment_column] == segment

        # Skip if no samples in this segment
        if segment_mask.sum() == 0:
            continue

        # Get segment name
        segment_name = (
            segment_values.get(segment, str(segment))
            if segment_values
            else str(segment)
        )

        # Get segment data
        X_segment = X[segment_mask]
        y_segment = y[segment_mask] if isinstance(y, pd.Series) else y[segment_mask]

        # Make predictions
        y_pred = model.predict(X_segment)

        # Calculate probabilities if model supports it
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_segment)[:, 1]
            metrics = calculate_metrics(y_segment, y_pred, y_prob)
        else:
            metrics = calculate_metrics(y_segment, y_pred)

        # Add churn rate
        metrics["churn_rate"] = y_segment.mean()
        metrics["count"] = len(y_segment)

        # Store metrics for this segment
        segment_metrics[segment_name] = metrics

        logger.info(
            f"Segment {segment_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, churn_rate={metrics['churn_rate']:.4f}, count={metrics['count']}"
        )

    return segment_metrics


def find_optimal_threshold(
    y_true: Union[List, np.ndarray], y_prob: Union[List, np.ndarray], metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold for classification.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy').

    Returns:
        Tuple: (optimal_threshold, optimal_metric_value)
    """
    logger.info(f"Finding optimal threshold for {metric}")

    # Get precision-recall curve data
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Add 1.0 to thresholds for the last point in the precision-recall curve
    thresholds = np.append(thresholds, 1.0)

    # Calculate metrics for different thresholds
    f1_scores = []
    balanced_accuracy_scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        f1_scores.append(metrics["f1"])
        balanced_accuracy_scores.append(metrics["balanced_accuracy"])

    # Choose metric to optimize
    if metric == "f1":
        scores = np.array(f1_scores)
    elif metric == "precision":
        scores = np.array(precision)
    elif metric == "recall":
        scores = np.array(recall)
    elif metric == "balanced_accuracy":
        scores = np.array(balanced_accuracy_scores)
    else:
        logger.warning(f"Unsupported metric: {metric}, using f1")
        scores = np.array(f1_scores)

    # Find optimal threshold
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    logger.info(
        f"Optimal threshold: {optimal_threshold:.4f}, {metric}: {optimal_score:.4f}"
    )

    return optimal_threshold, optimal_score

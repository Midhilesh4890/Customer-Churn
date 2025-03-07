"""
Cross-validation module for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer, confusion_matrix
)

from src.utils.logger import get_logger
from src.config import RANDOM_STATE

logger = get_logger(__name__)


def perform_kfold_cv(
    models_dict: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: Union[str, List[str], Dict[str, Callable]] = None,
    return_train_score: bool = False
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform k-fold cross-validation for multiple models.
    
    Args:
        models_dict: Dictionary of models to evaluate.
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of folds for cross-validation.
        scoring: Scoring metrics to evaluate.
        return_train_score: Whether to return training scores.
        
    Returns:
        Dict: Dictionary of cross-validation results by model.
    """
    logger.info(f"Performing {n_splits}-fold cross-validation")

    if scoring is None:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }

    # Create the k-fold cross-validation iterator
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)

    # Dictionary to store results
    cv_results = {}

    # Perform cross-validation for each model
    for model_name, model in models_dict.items():
        logger.info(f"Cross-validating {model_name} model")

        try:
            # Clone the model to ensure independence between folds
            model_clone = model.__class__(**model.get_params())

            # Perform cross-validation
            results = cross_validate(
                model_clone,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=-1
            )

            cv_results[model_name] = results

            # Calculate and log mean scores
            for metric in scoring.keys():
                test_scores = results[f'test_{metric}']
                mean_score = np.mean(test_scores)
                std_score = np.std(test_scores)
                logger.info(
                    f"{model_name} {n_splits}-fold CV {metric}: {mean_score:.4f} ± {std_score:.4f}")

        except Exception as e:
            logger.error(
                f"Error during cross-validation for {model_name}: {e}")

    return cv_results


def compare_cv_results(
    cv_results: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compare cross-validation results for multiple models.
    
    Args:
        cv_results: Dictionary of cross-validation results by model.
        metrics: List of metrics to compare.
        
    Returns:
        pandas.DataFrame: DataFrame with mean and std of CV scores.
    """
    if metrics is None:
        # Auto-detect metrics from the first model results
        model_name = next(iter(cv_results))
        metrics = [key.replace('test_', '') for key in cv_results[model_name].keys()
                   if key.startswith('test_')]

    # Create a list to store results
    comparison_rows = []

    for model_name, results in cv_results.items():
        for metric in metrics:
            test_key = f'test_{metric}'

            if test_key in results:
                test_scores = results[test_key]
                mean_score = np.mean(test_scores)
                std_score = np.std(test_scores)

                comparison_rows.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Mean': mean_score,
                    'Std': std_score
                })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_rows)

    return comparison_df


def get_optimal_threshold_by_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    thresholds: np.ndarray = None
) -> Tuple[float, float]:
    """
    Find the optimal threshold for a given metric.
    
    Args:
        y_true: True binary labels.
        y_prob: Probability estimates of the positive class.
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy').
        thresholds: Array of thresholds to evaluate (default: np.linspace(0.01, 0.99, 99)).
        
    Returns:
        Tuple: (optimal_threshold, optimal_metric_value)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = (specificity + sensitivity) / 2
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    return optimal_threshold, optimal_score


def analyze_threshold_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: List[str] = ['f1', 'precision', 'recall', 'balanced_accuracy'],
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Analyze performance metrics across different probability thresholds.
    
    Args:
        y_true: True binary labels.
        y_prob: Probability estimates of the positive class.
        metrics: List of metrics to evaluate.
        thresholds: Array of thresholds to evaluate (default: np.linspace(0.01, 0.99, 99)).
        
    Returns:
        pandas.DataFrame: DataFrame with metrics by threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (specificity + recall) / 2

        result = {
            'threshold': threshold,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy
        }

        results.append(result)

    return pd.DataFrame(results)

"""
Advanced model visualization utilities for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils.logger import get_logger
from src.config import VISUALIZATIONS_DIR

logger = get_logger(__name__)


def set_visualization_style():
    """
    Set the style for visualizations.
    """
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14


def plot_cv_results(
    cv_comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot cross-validation results comparison.

    Args:
        cv_comparison_df: DataFrame with cross-validation results comparison.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting cross-validation results comparison")

    set_visualization_style()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot model comparison with error bars
    sns.barplot(x="Model", y="Mean", hue="Metric", data=cv_comparison_df, ax=ax)

    # Add error bars
    for i, row in enumerate(cv_comparison_df.itertuples()):
        model_count = len(cv_comparison_df["Model"].unique())
        metric_count = len(cv_comparison_df["Metric"].unique())

        # Calculate the x position for the error bar
        x_pos = (
            (i // metric_count)
            + (i % metric_count) / metric_count
            - 0.5 * (metric_count - 1) / metric_count
        )

        ax.errorbar(
            x=x_pos, y=row.Mean, yerr=row.Std, fmt="none", color="black", capsize=5
        )

    ax.set_title("Cross-Validation Results by Model and Metric")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cross-validation results plot to {output_path}")

    return fig


def plot_multiple_roc_curves(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true: True binary labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting multiple ROC curves")

    set_visualization_style()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve for each model
    for model_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

    # Plot random classifier
    ax.plot([0, 1], [0, 1], "k--", lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve Comparison")
    ax.legend(loc="lower right")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved multiple ROC curves plot to {output_path}")

    return fig


def plot_multiple_pr_curves(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot precision-recall curves for multiple models.

    Args:
        y_true: True binary labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting multiple precision-recall curves")

    set_visualization_style()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate baseline
    baseline = sum(y_true) / len(y_true)

    # Plot precision-recall curve for each model
    for model_name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, lw=2, label=f"{model_name} (AUC = {pr_auc:.3f})")

    # Plot baseline
    ax.axhline(
        y=baseline,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline (No Skill): {baseline:.3f}",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="best")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved multiple precision-recall curves plot to {output_path}")

    return fig


def plot_threshold_analysis(
    threshold_results: pd.DataFrame,
    metrics: List[str] = ["precision", "recall", "f1", "balanced_accuracy"],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot metrics across different probability thresholds.

    Args:
        threshold_results: DataFrame with metrics by threshold.
        metrics: List of metrics to plot.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting threshold analysis")

    set_visualization_style()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot metrics by threshold
    for metric in metrics:
        if metric in threshold_results.columns:
            ax.plot(
                threshold_results["threshold"],
                threshold_results[metric],
                lw=2,
                label=metric.capitalize(),
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics by Probability Threshold")
    ax.legend(loc="best")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved threshold analysis plot to {output_path}")

    return fig


def plot_optimal_thresholds(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    metrics: List[str] = ["f1", "precision", "recall", "balanced_accuracy"],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot optimal thresholds by metric for multiple models.

    Args:
        y_true: True binary labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        metrics: List of metrics to evaluate.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting optimal thresholds by metric")

    # Import here to avoid circular imports
    from src.models.cross_validation import analyze_threshold_performance

    set_visualization_style()

    # Calculate optimal thresholds
    thresholds = np.linspace(0.01, 0.99, 99)

    # Create figure with subplots
    n_models = len(y_prob_dict)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_models, n_metrics, figsize=figsize, sharex=True, sharey=True
    )

    # Adjust figure for single model or metric case
    if n_models == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = np.array([axes])
    elif n_metrics == 1:
        axes = axes.reshape(-1, 1)

    # Plot threshold curves for each model and metric
    for i, (model_name, y_prob) in enumerate(y_prob_dict.items()):
        # Analyze threshold performance
        threshold_results = analyze_threshold_performance(
            y_true, y_prob, metrics, thresholds
        )

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Plot metric curve
            ax.plot(
                threshold_results["threshold"],
                threshold_results[metric],
                lw=2,
                color="blue",
            )

            # Find optimal threshold
            optimal_idx = threshold_results[metric].idxmax()
            optimal_threshold = threshold_results.loc[optimal_idx, "threshold"]
            optimal_value = threshold_results.loc[optimal_idx, metric]

            # Mark optimal point
            ax.scatter(optimal_threshold, optimal_value, color="red", s=100, zorder=10)

            ax.axvline(x=optimal_threshold, color="red", linestyle="--", alpha=0.5)

            # Set title for top row only
            if i == 0:
                ax.set_title(metric.capitalize())

            # Set y-label for leftmost column only
            if j == 0:
                ax.set_ylabel(model_name)

            # Set x-label for bottom row only
            if i == n_models - 1:
                ax.set_xlabel("Threshold")

            # Add optimal threshold value as text
            ax.text(
                0.5,
                0.05,
                f"Optimal: {optimal_threshold:.2f}\nScore: {optimal_value:.3f}",
                ha="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )

            # Set axis limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved optimal thresholds plot to {output_path}")

    return fig


def plot_confusion_matrices_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot confusion matrices at different probability thresholds.

    Args:
        y_true: True binary labels.
        y_prob: Probability estimates of the positive class.
        thresholds: List of thresholds to evaluate.
        output_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting confusion matrices at different thresholds")

    from sklearn.metrics import confusion_matrix

    set_visualization_style()

    # Create figure with subplots
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, n_thresholds, figsize=figsize)

    # Adjust for single threshold case
    if n_thresholds == 1:
        axes = [axes]

    # Plot confusion matrix for each threshold
    for i, threshold in enumerate(thresholds):
        ax = axes[i]

        # Get predictions at this threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax
        )

        # Set title and labels
        ax.set_title(f"Threshold = {threshold:.2f}\nF1 = {f1:.3f}")
        ax.set_xlabel("Predicted Label")

        # Set y-label for leftmost subplot only
        if i == 0:
            ax.set_ylabel("True Label")

        # Set tick labels
        ax.set_xticklabels(["Not Churn", "Churn"])
        ax.set_yticklabels(["Not Churn", "Churn"])

        # Add metrics text
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}"
        )

        # Add text below the heatmap
        ax.text(
            0.5, -0.15, metrics_text, ha="center", transform=ax.transAxes, fontsize=10
        )

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrices at thresholds plot to {output_path}")

    return fig

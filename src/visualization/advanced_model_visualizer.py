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
from src.pipeline.base import PipelineComponent

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


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ["accuracy", "precision", "recall", "f1_score", "auc"],
    title: str = "Model Performance Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    sort_by: Optional[str] = "f1_score",
    ascending: bool = False,
) -> plt.Figure:
    """
    Create a comprehensive visualization comparing multiple models across different metrics.

    Args:
        metrics_dict: Dictionary with model names as keys and dictionaries of metrics as values
        metrics_to_plot: List of metric names to include in the visualization
        title: Title for the figure
        output_path: Path to save the figure
        figsize: Figure size as (width, height)
        sort_by: Metric to sort the models by (or None for no sorting)
        ascending: Whether to sort in ascending order

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Plotting model comparison visualization")

    # Set plot style
    set_visualization_style()

    # Prepare data for plotting
    data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name, value in metrics.items():
            if metric_name in metrics_to_plot and isinstance(value, (int, float)):
                data.append(
                    {
                        "Model": model_name,
                        "Metric": metric_name.replace("_score", "").capitalize(),
                        "Value": value,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Check if we have data to plot
    if df.empty:
        logger.warning("No valid metrics data found for comparison visualization")
        fig = plt.figure(figsize=figsize)
        plt.text(
            0.5,
            0.5,
            "No valid metrics data available for comparison",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    # Sort models if requested
    if (
        sort_by is not None
        and sort_by.replace("_score", "").capitalize() in df["Metric"].unique()
    ):
        sort_metric = sort_by.replace("_score", "").capitalize()
        model_order = (
            df[df["Metric"] == sort_metric]
            .sort_values("Value", ascending=ascending)["Model"]
            .unique()
        )
        # Create a categorical type with the desired order
        df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
        df = df.sort_values("Model")

    # Create figure with multiple plots
    fig = plt.figure(figsize=figsize)

    # 1. Main bar plot showing all metrics by model
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    sns.barplot(x="Model", y="Value", hue="Metric", data=df, ax=ax1)
    ax1.set_title(f"{title}", fontsize=16)
    ax1.set_xlabel("Model", fontsize=14)
    ax1.set_ylabel("Score", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # 2. Heatmap showing all metrics by model
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    pivot_df = df.pivot(index="Model", columns="Metric", values="Value")
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar=False, ax=ax2)
    ax2.set_title("Heatmap View", fontsize=14)

    # 3. Spider/Radar chart
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3, polar=True)

    # Prepare radar chart data
    metrics = df["Metric"].unique()
    N = len(metrics)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set up spider chart
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)

    # Add labels to the angles
    plt.xticks(angles[:-1], metrics)

    # Set y-limits
    ax3.set_ylim(0, 1)

    # Plot each model
    for model_name in df["Model"].unique():
        model_df = df[df["Model"] == model_name]
        values = []
        for metric in metrics:
            value_series = model_df[model_df["Metric"] == metric]["Value"]
            if len(value_series) > 0:
                values.append(value_series.values[0])
            else:
                values.append(0)  # Default value if metric is missing
        values += values[:1]  # Close the loop

        # Plot the model
        ax3.plot(angles, values, linewidth=2, label=model_name)
        ax3.fill(angles, values, alpha=0.1)

    ax3.set_title("Radar Comparison", fontsize=14)
    ax3.legend(loc="lower right", bbox_to_anchor=(1.2, 0))

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved model comparison visualization to {output_path}")

    return fig


class ModelComparisonComponent(PipelineComponent):
    """Component for generating model comparison visualization using real pipeline metrics."""

    def __init__(self, output_dir: Union[str, Path] = VISUALIZATIONS_DIR):
        """Initialize the model comparison component."""
        super().__init__(name="model_comparison")
        self.output_dir = Path(output_dir)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model comparison visualization using real metrics.

        Args:
            data: Dictionary containing data from previous pipeline stages,
                  including metrics from model evaluation.

        Returns:
            Dict: Dictionary with model comparison visualization added.
        """
        self.logger.info("Generating model comparison visualization with real metrics")

        # Try to locate metrics in the data
        metrics = data.get("metrics", {})

        # If not found directly, check inside model_pipeline results
        if not metrics and "model_pipeline" in data:
            model_pipeline_data = data["model_pipeline"]
            if isinstance(model_pipeline_data, dict):
                metrics = model_pipeline_data.get("metrics", {})

        # If still not found, look in the model_trainer component results
        if not metrics and "model_trainer" in data:
            metrics = data["model_trainer"].get("metrics", {})

        # Check if we found metrics to plot
        if not metrics:
            self.logger.warning("No metrics found for model comparison visualization")
            return data

        self.logger.info(f"Found metrics for {len(metrics)} models")

        # Create output directory if it doesn't exist
        comparison_dir = self.output_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate comparison visualization
            comparison_fig = plot_model_comparison(
                metrics,
                title="Churn Prediction Model Comparison",
                output_path=comparison_dir / "model_comparison.png",
                sort_by="f1_score",
            )

            # Create a new result dictionary with visualization added
            result = data.copy()
            result["model_comparison_fig"] = comparison_fig

            self.logger.info("Model comparison visualization generated successfully")

            return result

        except Exception as e:
            self.logger.error(f"Error generating model comparison visualization: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return data

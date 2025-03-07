import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

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


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting ROC curve")

    set_visualization_style()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve plot to {output_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting precision-recall curve")

    set_visualization_style()

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add baseline
    baseline = sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
               label=f'Baseline (No Skill): {baseline:.3f}')
    ax.legend()

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved precision-recall curve plot to {output_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        normalize: Whether to normalize the confusion matrix.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting confusion matrix")

    set_visualization_style()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        cbar=True,
        ax=ax
    )

    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # Set tick labels
    ax.set_xticklabels(['Not Churn', 'Churn'])
    ax.set_yticklabels(['Not Churn', 'Churn'])

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {output_path}")

    return fig


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_importance_df: DataFrame with feature importance.
        top_n: Number of top features to plot.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(f"Plotting top {top_n} feature importance")

    set_visualization_style()

    # Get the importance column name
    importance_col = [col for col in feature_importance_df.columns
                      if 'importance' in col.lower() or 'coefficient' in col.lower()][0]

    # Get top N features
    top_features = feature_importance_df.sort_values(
        importance_col, ascending=False
    ).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot feature importance
    sns.barplot(
        x=importance_col,
        y='feature',
        data=top_features.sort_values(importance_col),
        palette='viridis',
        ax=ax
    )

    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {output_path}")

    return fig


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = ['Not Churn', 'Churn'],
    max_depth: int = 3,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Plot decision tree visualization.
    
    Args:
        model: Trained decision tree model.
        feature_names: List of feature names.
        class_names: List of class names.
        max_depth: Maximum depth of the tree to plot.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(
        f"Plotting decision tree visualization (max_depth={max_depth})")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot decision tree
    plot_tree(
        model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        precision=2,
        ax=ax
    )

    ax.set_title(f'Decision Tree (Max Depth = {max_depth})')

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decision tree plot to {output_path}")

    return fig


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot model comparison.
    
    Args:
        metrics_dict: Dictionary of model metrics.
        metrics_to_plot: List of metrics to plot.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting model comparison")

    set_visualization_style()

    # Create dataframe for plotting
    model_metrics = []

    for model_name, metrics in metrics_dict.items():
        for metric, value in metrics.items():
            if metric in metrics_to_plot:
                model_metrics.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': value
                })

    model_metrics_df = pd.DataFrame(model_metrics)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot model comparison
    sns.barplot(
        x='Model',
        y='Value',
        hue='Metric',
        data=model_metrics_df,
        ax=ax
    )

    ax.set_title('Model Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.legend(title='Metric')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved model comparison plot to {output_path}")

    return fig


def plot_segment_performance(
    segment_metrics: Dict[str, Dict[str, float]],
    metric: str = 'f1',
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot model performance by segment.
    
    Args:
        segment_metrics: Dictionary of metrics by segment.
        metric: Metric to plot.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(f"Plotting model performance by segment (metric={metric})")

    set_visualization_style()

    # Create dataframe for plotting
    segment_data = []

    for segment, metrics in segment_metrics.items():
        if metric in metrics:
            segment_data.append({
                'Segment': segment,
                metric.capitalize(): metrics[metric],
                'Churn Rate': metrics.get('churn_rate', 0) * 100 if metrics.get('churn_rate', 0) <= 1 else metrics.get('churn_rate', 0),
                'Count': metrics.get('count', 0)
            })

    segment_df = pd.DataFrame(segment_data)

    # Sort by churn rate
    segment_df = segment_df.sort_values('Churn Rate', ascending=False)

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot metric bars
    sns.barplot(
        x='Segment',
        y=metric.capitalize(),
        data=segment_df,
        color='steelblue',
        ax=ax1
    )

    ax1.set_title(f'Model Performance ({metric.capitalize()}) by Segment')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel(f'{metric.capitalize()} Score')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Create second y-axis for churn rate
    ax2 = ax1.twinx()

    # Plot churn rate points
    ax2.plot(
        segment_df.index,
        segment_df['Churn Rate'],
        'ro-',
        linewidth=2,
        markersize=8
    )

    ax2.set_ylabel('Churn Rate (%)', color='r')
    ax2.tick_params(axis='y', colors='r')

    # Add count labels
    for i, count in enumerate(segment_df['Count']):
        ax1.text(
            i,
            0.05,
            f"n={count}",
            ha='center',
            va='bottom',
            color='gray',
            fontsize=9
        )

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved segment performance plot to {output_path}")

    return fig


def generate_model_visualizations(
    models_dict: Dict[str, BaseEstimator],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    metrics_dict: Dict[str, Dict[str, float]],
    feature_importance_dict: Dict[str, pd.DataFrame],
    segment_metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path = VISUALIZATIONS_DIR
) -> Dict[str, plt.Figure]:
    """
    Generate all model visualizations.
    
    Args:
        models_dict: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.
        metrics_dict: Dictionary of model metrics.
        feature_importance_dict: Dictionary of feature importance by model.
        segment_metrics_dict: Dictionary of metrics by segment by model.
        output_dir: Directory to save the plots.
        
    Returns:
        Dict: Dictionary of plot figures by plot name.
    """
    logger.info("Generating model visualizations")

    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Plot model comparison
    plots["model_comparison"] = plot_model_comparison(
        metrics_dict,
        output_path=model_dir / "model_comparison.png"
    )

    # Generate plots for each model
    for model_name, model in models_dict.items():
        logger.info(f"Generating visualizations for {model_name} model")

        model_specific_dir = model_dir / model_name
        model_specific_dir.mkdir(parents=True, exist_ok=True)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate probabilities if supported
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]

            # Plot ROC curve
            plots[f"{model_name}_roc_curve"] = plot_roc_curve(
                y_test, y_prob,
                output_path=model_specific_dir / "roc_curve.png"
            )

            # Plot precision-recall curve
            plots[f"{model_name}_pr_curve"] = plot_precision_recall_curve(
                y_test, y_prob,
                output_path=model_specific_dir / "pr_curve.png"
            )

        # Plot confusion matrix
        plots[f"{model_name}_confusion_matrix"] = plot_confusion_matrix(
            y_test, y_pred,
            output_path=model_specific_dir / "confusion_matrix.png"
        )

        # Plot feature importance if available
        if model_name in feature_importance_dict:
            plots[f"{model_name}_feature_importance"] = plot_feature_importance(
                feature_importance_dict[model_name],
                output_path=model_specific_dir / "feature_importance.png"
            )

        # Plot decision tree if it's a decision tree model
        if isinstance(model, DecisionTreeClassifier):
            plots[f"{model_name}_decision_tree"] = plot_decision_tree(
                model,
                feature_names=X_test.columns.tolist(),
                output_path=model_specific_dir / "decision_tree.png"
            )

        # Plot segment performance if available
        if model_name in segment_metrics_dict:
            plots[f"{model_name}_segment_performance"] = plot_segment_performance(
                segment_metrics_dict[model_name],
                output_path=model_specific_dir / "segment_performance.png"
            )

    logger.info(f"Generated {len(plots)} model visualizations")

    return plots

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def plot_target_distribution(
    df: pd.DataFrame,
    target_column: str = "Churn",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: Dataframe with the target column.
        target_column: Name of the target column.
        output_path: Path to save the plot (optional).
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(f"Plotting distribution of {target_column}")

    set_visualization_style()

    # Count target values
    target_counts = df[target_column].value_counts()

    # Calculate percentages
    target_percentages = (target_counts / len(df) * 100).round(1)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot counts
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax[0])
    ax[0].set_title(f"{target_column} Distribution (Counts)")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel(target_column)

    # Add count labels
    for i, count in enumerate(target_counts.values):
        ax[0].text(i, count + 50, f"{count}", ha="center")

    # Plot percentages
    sns.barplot(x=target_counts.index, y=target_percentages.values, ax=ax[1])
    ax[1].set_title(f"{target_column} Distribution (Percentage)")
    ax[1].set_ylabel("Percentage (%)")
    ax[1].set_xlabel(target_column)

    # Add percentage labels
    for i, pct in enumerate(target_percentages.values):
        ax[1].text(i, pct + 1, f"{pct}%", ha="center")

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_categorical_vs_target(
    df: pd.DataFrame,
    categorical_columns: List[str],
    target_column: str = "Churn",
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Dict[str, plt.Figure]:
    """
    Plot the relationship between categorical features and the target variable.
    
    Args:
        df: Dataframe with categorical columns and target column.
        categorical_columns: List of categorical column names.
        target_column: Name of the target column.
        output_dir: Directory to save the plots (optional).
        figsize: Figure size.
        
    Returns:
        Dict: Dictionary of plot figures by feature name.
    """
    logger.info(
        f"Plotting relationship between categorical features and {target_column}")

    set_visualization_style()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Filter categorical columns that exist in the dataframe
    valid_columns = [col for col in categorical_columns if col in df.columns]

    figures = {}

    for col in valid_columns:
        logger.info(f"Plotting {col} vs {target_column}")

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Get the counts
        counts = df.groupby([col, target_column]).size().unstack(fill_value=0)

        # Plot counts
        counts.plot(kind="bar", stacked=True, ax=ax[0])
        ax[0].set_title(f"{col} vs {target_column} (Counts)")
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel(col)
        ax[0].legend(title=target_column)

        # Plot percentages
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        percentages.plot(kind="bar", stacked=True, ax=ax[1])
        ax[1].set_title(f"{col} vs {target_column} (Percentage)")
        ax[1].set_ylabel("Percentage (%)")
        ax[1].set_xlabel(col)
        ax[1].legend(title=target_column)

        # Add percentage labels for the "Yes" (churn) category
        if "Yes" in percentages.columns:
            for i, (_, pct) in enumerate(percentages["Yes"].items()):
                ax[1].text(i, pct / 2, f"{pct:.1f}%",
                           ha="center", color="white")

        plt.tight_layout()

        # Save plot if output directory is provided
        if output_dir:
            output_path = output_dir / f"{col}_vs_{target_column}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")

        figures[col] = fig

    return figures


def plot_numerical_vs_target(
    df: pd.DataFrame,
    numerical_columns: List[str],
    target_column: str = "Churn",
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Dict[str, plt.Figure]:
    """
    Plot the relationship between numerical features and the target variable.
    
    Args:
        df: Dataframe with numerical columns and target column.
        numerical_columns: List of numerical column names.
        target_column: Name of the target column.
        output_dir: Directory to save the plots (optional).
        figsize: Figure size.
        
    Returns:
        Dict: Dictionary of plot figures by feature name.
    """
    logger.info(
        f"Plotting relationship between numerical features and {target_column}")

    set_visualization_style()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Filter numerical columns that exist in the dataframe
    valid_columns = [col for col in numerical_columns if col in df.columns]

    figures = {}

    for col in valid_columns:
        logger.info(f"Plotting {col} vs {target_column}")

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Box plot
        sns.boxplot(x=target_column, y=col, data=df, ax=ax[0])
        ax[0].set_title(f"{col} vs {target_column} (Box Plot)")
        ax[0].set_ylabel(col)
        ax[0].set_xlabel(target_column)

        # Violin plot
        sns.violinplot(x=target_column, y=col, data=df, ax=ax[1])
        ax[1].set_title(f"{col} vs {target_column} (Violin Plot)")
        ax[1].set_ylabel(col)
        ax[1].set_xlabel(target_column)

        plt.tight_layout()

        # Save plot if output directory is provided
        if output_dir:
            output_path = output_dir / f"{col}_vs_{target_column}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")

        figures[col] = fig

    return figures


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Plot correlation matrix of features.
    
    Args:
        df: Dataframe with features.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info("Plotting correlation matrix")

    set_visualization_style()

    # Convert categorical columns to numeric for correlation calculation
    df_numeric = df.copy()

    for col in df_numeric.columns:
        if df_numeric[col].dtype == "object" or df_numeric[col].dtype.name == "category":
            df_numeric[col] = pd.factorize(df_numeric[col])[0]

    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Feature Correlation Matrix")

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_tenure_distribution(
    df: pd.DataFrame,
    tenure_column: str = "tenure",
    target_column: str = "Churn",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot the distribution of tenure by churn status.
    
    Args:
        df: Dataframe with tenure and target columns.
        tenure_column: Name of the tenure column.
        target_column: Name of the target column.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(f"Plotting tenure distribution by {target_column}")

    set_visualization_style()

    # Create tenure bins
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']

    # Add tenure group column
    df_with_groups = df.copy()
    df_with_groups['tenure_group'] = pd.cut(
        df_with_groups[tenure_column], bins=bins, labels=labels, right=True
    )

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot tenure distribution
    sns.histplot(
        data=df, x=tenure_column, hue=target_column,
        multiple="stack", bins=12, ax=ax[0]
    )
    ax[0].set_title(f"Distribution of {tenure_column} by {target_column}")
    ax[0].set_xlabel(f"{tenure_column} (months)")
    ax[0].set_ylabel("Count")

    # Plot churn rate by tenure group
    group_churn = df_with_groups.groupby('tenure_group')[target_column].apply(
        lambda x: (x == 'Yes').mean() * 100 if 'Yes' in x.unique() else 0
    ).reset_index()

    sns.barplot(x='tenure_group', y=target_column, data=group_churn, ax=ax[1])
    ax[1].set_title(f"{target_column} Rate by Tenure Group")
    ax[1].set_xlabel("Tenure Group (months)")
    ax[1].set_ylabel(f"{target_column} Rate (%)")

    # Add percentage labels
    for i, rate in enumerate(group_churn[target_column]):
        ax[1].text(i, rate + 1, f"{rate:.1f}%", ha="center")

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_service_distribution(
    df: pd.DataFrame,
    service_columns: List[str],
    target_column: str = "Churn",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot churn rate by service subscription.
    
    Args:
        df: Dataframe with service columns and target column.
        service_columns: List of service column names.
        target_column: Name of the target column.
        output_path: Path to save the plot (optional).
        figsize: Figure size.
        
    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    logger.info(f"Plotting {target_column} rate by service subscription")

    set_visualization_style()

    # Filter service columns that exist in the dataframe
    valid_columns = [col for col in service_columns if col in df.columns]

    # Calculate churn rate by service
    service_churn = []

    for col in valid_columns:
        # Skip if column doesn't have "Yes" values (e.g., if it's not a binary service column)
        if 'Yes' not in df[col].unique():
            continue

        churn_rate_with_service = df[df[col] == 'Yes'][target_column].apply(
            lambda x: 1 if x == 'Yes' else 0
        ).mean() * 100

        churn_rate_without_service = df[df[col] != 'Yes'][target_column].apply(
            lambda x: 1 if x == 'Yes' else 0
        ).mean() * 100

        service_churn.append({
            'Service': col,
            'With Service': churn_rate_with_service,
            'Without Service': churn_rate_without_service
        })

    # Create a dataframe for plotting
    service_churn_df = pd.DataFrame(service_churn)

    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        service_churn_df,
        id_vars=['Service'],
        value_vars=['With Service', 'Without Service'],
        var_name='Status',
        value_name='Churn Rate (%)'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot churn rate by service
    sns.barplot(x='Service', y='Churn Rate (%)',
                hue='Status', data=melted_df, ax=ax)
    ax.set_title(f"{target_column} Rate by Service Subscription")
    ax.set_xlabel("Service")
    ax.set_ylabel(f"{target_column} Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.text(
            p.get_x() + p.get_width() / 2.,
            p.get_height() + 0.5,
            f"{p.get_height():.1f}%",
            ha="center"
        )

    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def generate_eda_visualizations(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numerical_columns: List[str],
    service_columns: List[str],
    target_column: str = "Churn",
    output_dir: Path = VISUALIZATIONS_DIR
) -> Dict[str, plt.Figure]:
    """
    Generate all EDA visualizations.
    
    Args:
        df: Dataframe with features and target column.
        categorical_columns: List of categorical column names.
        numerical_columns: List of numerical column names.
        service_columns: List of service column names.
        target_column: Name of the target column.
        output_dir: Directory to save the plots.
        
    Returns:
        Dict: Dictionary of plot figures by plot name.
    """
    logger.info("Generating EDA visualizations")

    eda_dir = Path(output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Plot target distribution
    plots["target_distribution"] = plot_target_distribution(
        df, target_column, eda_dir / "target_distribution.png"
    )

    # Plot categorical features vs target
    categorical_plots = plot_categorical_vs_target(
        df, categorical_columns, target_column, eda_dir / "categorical"
    )
    plots.update({f"categorical_{key}": value for key,
                 value in categorical_plots.items()})

    # Plot numerical features vs target
    numerical_plots = plot_numerical_vs_target(
        df, numerical_columns, target_column, eda_dir / "numerical"
    )
    plots.update({f"numerical_{key}": value for key,
                 value in numerical_plots.items()})

    # Plot correlation matrix
    plots["correlation_matrix"] = plot_correlation_matrix(
        df, eda_dir / "correlation_matrix.png"
    )

    # Plot tenure distribution
    plots["tenure_distribution"] = plot_tenure_distribution(
        df, "tenure", target_column, eda_dir / "tenure_distribution.png"
    )

    # Plot service distribution
    plots["service_distribution"] = plot_service_distribution(
        df, service_columns, target_column, eda_dir / "service_distribution.png"
    )

    logger.info(f"Generated {len(plots)} EDA visualizations")

    return plots

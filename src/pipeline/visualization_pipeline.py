"""
Visualization pipeline components for the churn prediction project.
"""
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from src.pipeline.base import PipelineComponent
from src.visualization.eda_visualizer import generate_eda_visualizations
from src.visualization.advanced_model_visualizer import (
    plot_cv_results,
    plot_multiple_roc_curves,
    plot_multiple_pr_curves,
    plot_threshold_analysis,
    plot_optimal_thresholds,
    plot_confusion_matrices_at_thresholds,
    ModelComparisonComponent,
)
from src.config import (
    VISUALIZATIONS_DIR,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
)


class EDAVisualizationComponent(PipelineComponent):
    """
    Component for generating exploratory data analysis visualizations.
    """

    def __init__(
        self,
        categorical_columns: List[str] = CATEGORICAL_FEATURES,
        numerical_columns: List[str] = NUMERICAL_FEATURES,
        service_columns: List[str] = None,
        target_column: str = TARGET,
        output_dir: Union[str, Path] = VISUALIZATIONS_DIR,
    ):
        """
        Initialize the EDA visualization component.

        Args:
            categorical_columns: List of categorical column names.
            numerical_columns: List of numerical column names.
            service_columns: List of service column names.
            target_column: Name of the target column.
            output_dir: Directory to save the plots.
        """
        super().__init__(name="eda_visualization")
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        if service_columns is None:
            self.service_columns = [
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
        else:
            self.service_columns = service_columns

        self.target_column = target_column
        self.output_dir = Path(output_dir)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate EDA visualizations.

        Args:
            data: Dictionary containing cleaned data.

        Returns:
            Dict: Dictionary with EDA visualization results added.
        """
        self.logger.info("Generating EDA visualizations")

        # First try to get the cleaned data directly
        df_cleaned = data.get("data_cleaner")

        # If not found, try to retrieve it from the data pipeline results
        if df_cleaned is None and "data_pipeline" in data:
            pipeline_data = data["data_pipeline"]
            if isinstance(pipeline_data, dict):
                df_cleaned = pipeline_data.get("data_cleaner")

        # If still not found, try to use the original raw data
        if df_cleaned is None:
            df_cleaned = data.get("data_loader")
            if df_cleaned is not None:
                self.logger.info("Using raw data for EDA visualizations")

        # Try to find data from the "models" stage which might have the original data
        if df_cleaned is None and "models" in data:
            df_cleaned = data.get("X_test_original")
            self.logger.info("Using X_test_original for EDA visualizations")

        # Check if we have any DataFrames in the data dictionary
        if df_cleaned is None:
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    df_cleaned = value
                    self.logger.info(f"Using {key} for EDA visualizations")
                    break

        # As a last resort, check if we received the original DataLoader component output
        if df_cleaned is None and isinstance(data, pd.DataFrame):
            df_cleaned = data
            self.logger.info("Using DataFrame passed directly for EDA visualizations")

        # If we still don't have data, log an error and return
        if df_cleaned is None:
            self.logger.error("No suitable data found for EDA visualizations")
            return data

        # Check if the target column exists in the DataFrame
        if self.target_column not in df_cleaned.columns:
            # Try to find an alternative target column
            alternative_targets = [
                "Churn_Binary",
                "target",
                "Target",
                "label",
                "Label",
                "y",
            ]
            for alt_target in alternative_targets:
                if alt_target in df_cleaned.columns:
                    self.logger.info(
                        f"Target column '{self.target_column}' not found, using '{alt_target}' instead"
                    )
                    self.target_column = alt_target
                    break
            else:
                # If we get here, none of the alternative targets were found
                self.logger.error(
                    f"Target column '{self.target_column}' not found in data"
                )

                # Try to get data from the raw cleaned data which should have the target
                original_data = data.get("data_pipeline", {})
                if isinstance(original_data, dict):
                    original_df = original_data.get("data_cleaner")
                    if (
                        isinstance(original_df, pd.DataFrame)
                        and self.target_column in original_df.columns
                    ):
                        df_cleaned = original_df
                        self.logger.info(
                            f"Using original cleaned data with target column '{self.target_column}'"
                        )
                    else:
                        return data
                else:
                    return data

        try:
            # Generate EDA visualizations
            eda_plots = generate_eda_visualizations(
                df_cleaned,
                self.categorical_columns,
                self.numerical_columns,
                self.service_columns,
                self.target_column,
                self.output_dir,
            )

            # Create a new result dictionary with EDA visualization results
            result = data.copy()
            result["eda_plots"] = eda_plots

            return result

        except Exception as e:
            self.logger.error(f"Error generating EDA visualizations: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return data


class ModelVisualizationComponent(PipelineComponent):
    """
    Component for generating model performance visualizations.
    """

    def __init__(self, output_dir: Union[str, Path] = VISUALIZATIONS_DIR):
        """
        Initialize the model visualization component.

        Args:
            output_dir: Directory to save the plots.
        """
        super().__init__(name="model_visualization")
        self.output_dir = Path(output_dir) / "advanced"

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model performance visualizations.

        Args:
            data: Dictionary containing model evaluation results.

        Returns:
            Dict: Dictionary with model visualization results added.
        """
        self.logger.info("Generating model performance visualizations")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get the required data, looking in both direct keys and nested results
        y_prob_dict = data.get("y_prob_dict", {})
        y_test = data.get("y_test")
        cv_comparison_df = data.get("cv_comparison_df")
        threshold_results = data.get("threshold_results", {})

        # Try to find data in model pipeline results if not found directly
        if "model_pipeline" in data:
            model_data = data["model_pipeline"]
            if isinstance(model_data, dict):
                if not y_prob_dict and "y_prob_dict" in model_data:
                    y_prob_dict = model_data["y_prob_dict"]
                if y_test is None and "y_test" in model_data:
                    y_test = model_data["y_test"]

        # Try to find data in evaluation pipeline results if not found directly
        if "evaluation_pipeline" in data:
            eval_data = data["evaluation_pipeline"]
            if isinstance(eval_data, dict):
                if not y_prob_dict and "y_prob_dict" in eval_data:
                    y_prob_dict = eval_data["y_prob_dict"]
                if y_test is None and "y_test" in eval_data:
                    y_test = eval_data["y_test"]
                if cv_comparison_df is None and "cv_comparison_df" in eval_data:
                    cv_comparison_df = eval_data["cv_comparison_df"]
                if not threshold_results and "threshold_results" in eval_data:
                    threshold_results = eval_data["threshold_results"]

        # Dictionary to store visualization results
        visualization_results = {}

        # Plot cross-validation results if available
        if cv_comparison_df is not None:
            try:
                cv_plot = plot_cv_results(
                    cv_comparison_df,
                    output_path=self.output_dir / "cross_validation_results.png",
                )
                visualization_results["cv_plot"] = cv_plot
                self.logger.info("Generated cross-validation results plot")
            except Exception as e:
                self.logger.error(f"Error plotting cross-validation results: {e}")
        else:
            self.logger.warning(
                "Cross-validation results not available for visualization"
            )

        # Plot ROC and PR curves if probability predictions are available
        if y_prob_dict and y_test is not None:
            try:
                # Plot multiple ROC curves
                roc_plot = plot_multiple_roc_curves(
                    y_test,
                    y_prob_dict,
                    output_path=self.output_dir / "roc_curves_comparison.png",
                )
                visualization_results["roc_plot"] = roc_plot
                self.logger.info("Generated ROC curves comparison plot")

                # Plot multiple PR curves
                pr_plot = plot_multiple_pr_curves(
                    y_test,
                    y_prob_dict,
                    output_path=self.output_dir / "pr_curves_comparison.png",
                )
                visualization_results["pr_plot"] = pr_plot
                self.logger.info("Generated PR curves comparison plot")

                # Plot optimal thresholds comparison
                try:
                    thresholds_plot = plot_optimal_thresholds(
                        y_test,
                        y_prob_dict,
                        output_path=self.output_dir
                        / "optimal_thresholds_comparison.png",
                    )
                    visualization_results["thresholds_plot"] = thresholds_plot
                    self.logger.info("Generated optimal thresholds comparison plot")
                except Exception as e:
                    self.logger.error(f"Error plotting optimal thresholds: {e}")
            except Exception as e:
                self.logger.error(f"Error plotting ROC/PR curves: {e}")
        else:
            self.logger.warning(
                "Probability predictions not available for visualization"
            )

        # Plot threshold analysis for each model
        for model_name, model_threshold_results in threshold_results.items():
            try:
                # Create model-specific directory
                model_dir = self.output_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                # Plot threshold analysis
                threshold_plot = plot_threshold_analysis(
                    model_threshold_results,
                    output_path=model_dir / "threshold_analysis.png",
                )
                visualization_results[f"{model_name}_threshold_plot"] = threshold_plot
                self.logger.info(f"Generated threshold analysis plot for {model_name}")

                # Plot confusion matrices at different thresholds
                if y_test is not None and model_name in y_prob_dict:
                    cm_plot = plot_confusion_matrices_at_thresholds(
                        y_test,
                        y_prob_dict[model_name],
                        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                        output_path=model_dir / "confusion_matrices_at_thresholds.png",
                    )
                    visualization_results[f"{model_name}_cm_plot"] = cm_plot
                    self.logger.info(
                        f"Generated confusion matrices plot for {model_name}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error plotting threshold analysis for {model_name}: {e}"
                )

        # Create a new result dictionary with visualization results
        result = data.copy()
        result["visualization_results"] = visualization_results

        return result


class VisualizationPipeline(PipelineComponent):
    """
    End-to-end visualization pipeline that combines EDA and model visualizations.
    """

    def __init__(
        self,
        run_eda: bool = True,
        categorical_columns: List[str] = CATEGORICAL_FEATURES,
        numerical_columns: List[str] = NUMERICAL_FEATURES,
        service_columns: List[str] = None,
        target_column: str = TARGET,
        output_dir: Union[str, Path] = VISUALIZATIONS_DIR,
    ):
        """
        Initialize the visualization pipeline.

        Args:
            run_eda: Whether to run EDA visualizations.
            categorical_columns: List of categorical column names.
            numerical_columns: List of numerical column names.
            service_columns: List of service column names.
            target_column: Name of the target column.
            output_dir: Directory to save the plots.
        """
        super().__init__(name="visualization_pipeline")

        # Create individual components
        self.components = []

        if run_eda:
            self.eda = EDAVisualizationComponent(
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                service_columns=service_columns,
                target_column=target_column,
                output_dir=output_dir,
            )
            self.components.append(self.eda)

        self.model_viz = ModelVisualizationComponent(output_dir=output_dir)
        self.components.append(self.model_viz)

        # Add the new model comparison component
        self.model_comparison = ModelComparisonComponent(output_dir=output_dir)
        self.components.append(self.model_comparison)

    @PipelineComponent.log_execution_time
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete visualization pipeline.

        Args:
            data: Dictionary containing data from previous pipelines.

        Returns:
            Dict: Dictionary with visualization results.
        """
        result = data.copy()

        try:
            # Run each component sequentially
            for component in self.components:
                self.logger.info(f"Running {component.name} component")
                result = component.run(result)

            return result
        except Exception as e:
            self.logger.error(f"Error in visualization pipeline: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return data

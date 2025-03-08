import time
import argparse
from pathlib import Path

from src.pipeline.main_pipeline import create_churn_pipeline
from src.utils.logger import get_logger
from src.config import RAW_DATA_PATH

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to raw data file",
    )

    parser.add_argument(
        "--skip_eda", action="store_true", help="Skip exploratory data analysis"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["lr", "dt", "rf", "ensemble", "all"],
        default=["all"],
        help="Models to train",
    )

    parser.add_argument("--skip_cv", action="store_true",
                        help="Skip cross-validation")

    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--skip_threshold", action="store_true", help="Skip threshold analysis"
    )

    parser.add_argument(
        "--skip_saving_models", action="store_true", help="Skip saving models to disk"
    )

    parser.add_argument(
        "--skip_saving_data", action="store_true", help="Skip saving processed data to disk"
    )

    return parser.parse_args()


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

    # Create the pipeline
    pipeline = create_churn_pipeline(
        data_path=args.data_path,
        models_to_train=models_to_train,
        run_eda=not args.skip_eda,
        run_cv=not args.skip_cv,
        cv_folds=args.cv_folds,
        run_threshold_opt=not args.skip_threshold,
        save_models=not args.skip_saving_models,
        save_output=not args.skip_saving_data
    )

    # Run the pipeline
    results = pipeline.run()

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(
        f"Pipeline execution completed in {execution_time:.2f} seconds")

    # Return the results for potential further processing
    return results


if __name__ == "__main__":
    main()

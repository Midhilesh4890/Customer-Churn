import os
from pathlib import Path

# Define base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"

# Data processing settings
RANDOM_STATE = 42
TEST_SIZE = 0.3
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
ID_COLUMN = "customerID"

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced"
}

DECISION_TREE_PARAMS = {
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced"
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1
}

# Feature engineering settings
SELECTED_FEATURES = [
    "Contract",
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
    "MonthlyCharges",
    "SeniorCitizen",
    "MultipleLines",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies"
]

"""
Logistic Regression model implementation for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

from src.utils.logger import get_logger
from src.config import RANDOM_STATE, LOGISTIC_REGRESSION_PARAMS

logger = get_logger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Parameters for the model.
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model.
    """
    logger.info("Training Logistic Regression model")

    if params is None:
        params = LOGISTIC_REGRESSION_PARAMS

    # Create and train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    logger.info("Logistic Regression model training completed")

    return model


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]] = None,
    cv: int = 5
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Tune hyperparameters for Logistic Regression model using grid search.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Grid of parameters to search.
        cv: Number of cross-validation folds.
        
    Returns:
        Tuple: (best_model, best_params)
    """
    logger.info("Tuning Logistic Regression hyperparameters")

    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        }

    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    # Create base model
    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

    # Create cross-validation strategy
    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Create grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        refit='f1',
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.info(f"Best parameters found: {best_params}")
    logger.info(
        f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    return best_model, best_params


def get_logistic_regression_feature_importance(
    model: LogisticRegression,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from Logistic Regression model.
    
    Args:
        model: Trained Logistic Regression model.
        feature_names: List of feature names.
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance.
    """
    logger.info("Extracting feature importance from Logistic Regression model")

    # Get coefficients
    coef = model.coef_[0]

    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'absolute_importance': np.abs(coef)
    })

    # Sort by absolute importance
    feature_importance = feature_importance.sort_values(
        'absolute_importance', ascending=False)

    logger.info("Feature importance extracted successfully")

    return feature_importance

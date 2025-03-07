"""
Decision Tree model implementation for the churn prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

from src.utils.logger import get_logger
from src.config import RANDOM_STATE, DECISION_TREE_PARAMS

logger = get_logger(__name__)


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Parameters for the model.
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained model.
    """
    logger.info("Training Decision Tree model")

    if params is None:
        params = DECISION_TREE_PARAMS

    # Create and train model
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Decision Tree model training completed")

    return model


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]] = None,
    cv: int = 5
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Tune hyperparameters for Decision Tree model using grid search.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Grid of parameters to search.
        cv: Number of cross-validation folds.
        
    Returns:
        Tuple: (best_model, best_params)
    """
    logger.info("Tuning Decision Tree hyperparameters")

    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
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
    base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)

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


def get_decision_tree_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from Decision Tree model.
    
    Args:
        model: Trained Decision Tree model.
        feature_names: List of feature names.
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance.
    """
    logger.info("Extracting feature importance from Decision Tree model")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False)

    logger.info("Feature importance extracted successfully")

    return feature_importance


def export_decision_tree_visualization(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = ['Not Churn', 'Churn'],
    output_file: str = 'decision_tree.dot'
) -> str:
    """
    Export the decision tree visualization to a DOT file.
    
    Args:
        model: Trained Decision Tree model.
        feature_names: List of feature names.
        class_names: List of class names.
        output_file: Name of the output file.
        
    Returns:
        str: Path to the output file.
    """
    logger.info(f"Exporting Decision Tree visualization to {output_file}")

    # Export tree to DOT format
    export_graphviz(
        model,
        out_file=output_file,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # Limit depth for visualization
    )

    logger.info("Decision Tree visualization exported successfully")

    return output_file

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Union

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

from src.utils.logger import get_logger
from src.config import RANDOM_STATE, RANDOM_FOREST_PARAMS
from src.models.logistic_regression import train_logistic_regression
from src.models.decision_tree import train_decision_tree

logger = get_logger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Parameters for the model.
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model.
    """
    logger.info("Training Random Forest model")

    if params is None:
        params = RANDOM_FOREST_PARAMS

    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Random Forest model training completed")

    return model


def train_voting_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimators: List[Tuple[str, BaseEstimator]] = None,
    voting: str = 'soft'
) -> VotingClassifier:
    """
    Train a Voting Classifier ensemble model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        estimators: List of (name, estimator) tuples.
        voting: Voting type ('hard' or 'soft').
        
    Returns:
        sklearn.ensemble.VotingClassifier: Trained model.
    """
    logger.info(f"Training Voting Classifier with {voting} voting")

    if estimators is None:
        # Train base models
        lr = train_logistic_regression(X_train, y_train)
        dt = train_decision_tree(X_train, y_train)
        rf = train_random_forest(X_train, y_train)

        estimators = [
            ('lr', lr),
            ('dt', dt),
            ('rf', rf)
        ]

    # Create and train voting classifier
    model = VotingClassifier(estimators=estimators, voting=voting)
    model.fit(X_train, y_train)

    logger.info("Voting Classifier training completed")

    return model


def train_adaboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_estimator: BaseEstimator = None,
    n_estimators: int = 50,
    learning_rate: float = 1.0
) -> AdaBoostClassifier:
    """
    Train an AdaBoost ensemble model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        base_estimator: Base estimator for AdaBoost.
        n_estimators: Number of estimators.
        learning_rate: Learning rate.
        
    Returns:
        sklearn.ensemble.AdaBoostClassifier: Trained model.
    """
    logger.info("Training AdaBoost model")

    if base_estimator is None:
        # Use decision tree with limited depth as base estimator
        base_estimator = DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE
        )

    # Create and train model
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    logger.info("AdaBoost model training completed")

    return model


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]] = None,
    cv: int = 5
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Tune hyperparameters for Random Forest model using grid search.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Grid of parameters to search.
        cv: Number of cross-validation folds.
        
    Returns:
        Tuple: (best_model, best_params)
    """
    logger.info("Tuning Random Forest hyperparameters")

    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
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
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

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


def get_ensemble_feature_importance(
    model: Union[RandomForestClassifier, VotingClassifier, AdaBoostClassifier],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from ensemble model.
    
    Args:
        model: Trained ensemble model.
        feature_names: List of feature names.
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance.
    """
    logger.info(
        f"Extracting feature importance from {type(model).__name__} model")

    if isinstance(model, RandomForestClassifier) or isinstance(model, AdaBoostClassifier):
        # Get feature importances directly
        importances = model.feature_importances_

        # Create DataFrame with feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False)

    elif isinstance(model, VotingClassifier):
        # For voting classifier, average importances from base estimators that support it
        importances = np.zeros(len(feature_names))
        count = 0

        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances += estimator.feature_importances_
                count += 1
            elif hasattr(estimator, 'coef_'):
                importances += np.abs(estimator.coef_[0])
                count += 1

        if count > 0:
            importances /= count

            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })

            # Sort by importance
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False)
        else:
            logger.warning(
                "Could not extract feature importance from voting classifier")
            return pd.DataFrame()
    else:
        logger.warning(
            f"Model {type(model).__name__} does not support feature importance")
        return pd.DataFrame()

    logger.info("Feature importance extracted successfully")

    return feature_importance

"""
Model factory for the churn prediction project.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.utils.logger import get_logger
from src.config import (
    MODELS_DIR,
    LOGISTIC_REGRESSION_PARAMS,
    DECISION_TREE_PARAMS,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE
)

logger = get_logger(__name__)


def create_model(model_type: str, params: Dict[str, Any] = None) -> BaseEstimator:
    """
    Create a model of the specified type with the given parameters.
    
    Args:
        model_type: Type of model to create ('lr', 'dt', 'rf', or 'ensemble').
        params: Parameters for the model.
        
    Returns:
        sklearn.base.BaseEstimator: The created model.
    """
    logger.info(f"Creating {model_type} model")

    if model_type == 'lr':
        # Logistic Regression
        if params is None:
            params = LOGISTIC_REGRESSION_PARAMS
        return LogisticRegression(**params)

    elif model_type == 'dt':
        # Decision Tree
        if params is None:
            params = DECISION_TREE_PARAMS
        return DecisionTreeClassifier(**params)

    elif model_type == 'rf':
        # Random Forest
        if params is None:
            params = RANDOM_FOREST_PARAMS
        return RandomForestClassifier(**params)

    elif model_type == 'ensemble':
        # Ensemble (Voting Classifier)
        models = []

        # Create base models
        lr = create_model('lr')
        dt = create_model('dt')
        rf = create_model('rf')

        models = [
            ('lr', lr),
            ('dt', dt),
            ('rf', rf)
        ]

        # Create voting classifier
        voting = 'soft' if params and params.get(
            'voting') == 'soft' else 'hard'

        return VotingClassifier(
            estimators=models,
            voting=voting
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> BaseEstimator:
    """
    Train the given model on the training data.
    
    Args:
        model: Model to train.
        X_train: Training features.
        y_train: Training labels.
        
    Returns:
        sklearn.base.BaseEstimator: The trained model.
    """
    logger.info(f"Training {type(model).__name__} model")

    # Train the model
    model.fit(X_train, y_train)

    logger.info(f"Model training completed")

    return model


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate the performance of the model on the test data.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dict: Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating {type(model).__name__} model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': auc
    }

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics.update({
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    })

    logger.info(f"Model evaluation results: {metrics}")

    return metrics


def get_feature_importance(
    model: BaseEstimator,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from the model.
    
    Args:
        model: Trained model.
        feature_names: List of feature names.
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance.
    """
    logger.info(
        f"Extracting feature importance from {type(model).__name__} model")

    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    elif isinstance(model, VotingClassifier):
        # For voting classifier, use the first model with feature importance
        for _, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                break
            elif hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_[0])
                break
        else:
            logger.warning(
                "Could not extract feature importance from voting classifier")
            return pd.DataFrame()
    else:
        logger.warning(
            f"Model {type(model).__name__} does not support feature importance")
        return pd.DataFrame()

    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False)

    logger.info(f"Feature importance extracted successfully")

    return feature_importance


def save_model(
    model: BaseEstimator,
    model_name: str,
    output_dir: Union[str, Path] = MODELS_DIR
) -> Path:
    """
    Save the model to disk.
    
    Args:
        model: Trained model to save.
        model_name: Name to use for the saved model file.
        output_dir: Directory to save the model file.
        
    Returns:
        pathlib.Path: Path to the saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create file path
    model_path = output_dir / f"{model_name}.joblib"

    logger.info(f"Saving model to {model_path}")

    # Save the model
    joblib.dump(model, model_path)

    logger.info(f"Model saved successfully")

    return model_path


def load_model(model_path: Union[str, Path]) -> BaseEstimator:
    """
    Load a model from disk.
    
    Args:
        model_path: Path to the saved model file.
        
    Returns:
        sklearn.base.BaseEstimator: The loaded model.
    """
    model_path = Path(model_path)

    logger.info(f"Loading model from {model_path}")

    # Load the model
    model = joblib.load(model_path)

    logger.info(f"Model {type(model).__name__} loaded successfully")

    return model

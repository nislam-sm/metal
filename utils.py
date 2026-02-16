"""
Utility Functions

Helper functions for metrics, validation, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    metrics : dict
        Dictionary containing RMSE, MAE, R², and Pearson correlation.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': corr,
        'n_samples': len(y_true)
    }


def validate_predictions(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    name: str = "Predictions"
) -> None:
    """
    Validate prediction matrix and target vector.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_models)
        Prediction matrix.
    
    y : array-like of shape (n_samples,)
        Target values.
    
    name : str
        Name for error messages.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
    
    y = np.asarray(y).ravel()
    
    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"{name}: X must be 2-dimensional, got shape {X.shape}")
    
    if len(y.shape) != 1:
        raise ValueError(f"{name}: y must be 1-dimensional, got shape {y.shape}")
    
    # Check lengths match
    if X.shape[0] != len(y):
        raise ValueError(
            f"{name}: X and y have inconsistent lengths: "
            f"{X.shape[0]} vs {len(y)}"
        )
    
    # Check for NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError(f"{name}: X contains NaN or Inf values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError(f"{name}: y contains NaN or Inf values")
    
    # Check for constant predictions
    n_constant = np.sum(np.std(X, axis=0) < 1e-10)
    if n_constant > 0:
        print(f"Warning: {n_constant} models have near-constant predictions")


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty-print metrics dictionary.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to print.
    
    title : str
        Title for the metrics display.
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if key == 'n_samples':
            print(f"{key:15s}: {value:,}")
        else:
            print(f"{key:15s}: {value:.6f}")
    
    print(f"{'='*50}\n")


def compute_gini_coefficient(weights: np.ndarray) -> float:
    """
    Compute Gini coefficient of weight distribution.
    
    Gini = 0: perfect equality (all weights equal)
    Gini = 1: maximum inequality (one weight is 1, rest are 0)
    
    Parameters
    ----------
    weights : array-like
        Weight values.
    
    Returns
    -------
    gini : float
        Gini coefficient.
    """
    weights = np.abs(np.asarray(weights))
    
    if len(weights) == 0:
        return 0.0
    
    # Normalize to sum to 1
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    # Sort weights
    sorted_weights = np.sort(weights)
    
    # Compute cumulative weights
    n = len(weights)
    index = np.arange(1, n + 1)
    
    # Gini coefficient formula
    gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    return gini


def get_top_k_features(
    weights: np.ndarray,
    feature_names: list,
    k: int = 10,
    by: str = 'absolute'
) -> pd.DataFrame:
    """
    Get top-k features by weight magnitude.
    
    Parameters
    ----------
    weights : array-like
        Feature weights.
    
    feature_names : list
        Names of features.
    
    k : int, default=10
        Number of top features to return.
    
    by : str, default='absolute'
        Ranking criterion: 'absolute', 'positive', or 'negative'.
    
    Returns
    -------
    top_features : pd.DataFrame
        DataFrame with top features and their weights.
    """
    weights = np.asarray(weights)
    
    if len(weights) != len(feature_names):
        raise ValueError("weights and feature_names must have same length")
    
    if by == 'absolute':
        ranking = np.argsort(np.abs(weights))[::-1]
    elif by == 'positive':
        ranking = np.argsort(weights)[::-1]
    elif by == 'negative':
        ranking = np.argsort(weights)
    else:
        raise ValueError(f"Unknown ranking criterion: {by}")
    
    top_k = min(k, len(weights))
    top_indices = ranking[:top_k]
    
    return pd.DataFrame({
        'rank': range(1, top_k + 1),
        'feature': [feature_names[i] for i in top_indices],
        'weight': weights[top_indices],
        'abs_weight': np.abs(weights[top_indices])
    })


def compute_condition_number(X: np.ndarray) -> float:
    """
    Compute condition number of correlation matrix.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    
    Returns
    -------
    kappa : float
        Condition number κ(C) = σ_max(C) / σ_min(C)
    """
    X = np.asarray(X)
    
    # Compute correlation matrix
    C = np.corrcoef(X.T)
    
    # Compute condition number
    if C.shape[0] == 1:
        return 1.0
    
    kappa = np.linalg.cond(C)
    
    return kappa


def create_oof_matrix(
    predictions_dict: Dict[str, np.ndarray],
    model_names: list = None
) -> pd.DataFrame:
    """
    Create OOF prediction matrix from dictionary.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays.
    
    model_names : list, optional
        Ordered list of model names. If None, uses sorted keys.
    
    Returns
    -------
    oof_matrix : pd.DataFrame
        DataFrame with OOF predictions.
    """
    if model_names is None:
        model_names = sorted(predictions_dict.keys())
    
    # Validate all predictions have same length
    lengths = [len(predictions_dict[name]) for name in model_names]
    if len(set(lengths)) != 1:
        raise ValueError("All predictions must have same length")
    
    # Create DataFrame
    data = {name: predictions_dict[name] for name in model_names}
    oof_matrix = pd.DataFrame(data)
    
    return oof_matrix

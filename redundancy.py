"""
Redundancy Projection in Prediction Space

Implements Algorithm 2 from the paper: multi-metric de-duplication
to reduce effective rank and improve conditioning of meta-learning.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error


class RedundancyProjector(BaseEstimator, TransformerMixin):
    """
    Redundancy-aware projection operator that selects a representative 
    subset of base models by removing near-collinear predictors.
    
    Algorithm 2: Redundancy Projection via Joint Similarity Filtering
    
    Parameters
    ----------
    tau_corr : float, default=0.95
        Correlation threshold. Models with correlation >= tau_corr are 
        considered redundant.
    
    tau_mse : float, default=0.01
        MSE threshold for prediction similarity. Models must be both
        highly correlated AND prediction-wise similar to be removed.
    
    tau_var : float, default=0.01
        Variance threshold for low-information model removal.
        Models with variance < tau_var are pruned.
    
    verbose : int, default=1
        Verbosity level. 0 = silent, 1 = progress, 2 = detailed.
    
    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of retained models after projection.
    
    removed_models_ : list
        Information about removed models (index, reason, correlation, etc.).
    
    condition_number_before_ : float
        Condition number of correlation matrix before projection.
    
    condition_number_after_ : float
        Condition number of correlation matrix after projection.
    
    n_models_initial_ : int
        Number of models before projection.
    
    n_models_retained_ : int
        Number of models after projection.
    """
    
    def __init__(
        self,
        tau_corr: float = 0.95,
        tau_mse: float = 0.01,
        tau_var: float = 0.01,
        verbose: int = 1
    ):
        self.tau_corr = tau_corr
        self.tau_mse = tau_mse
        self.tau_var = tau_var
        self.verbose = verbose
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'RedundancyProjector':
        """
        Learn which models to retain based on redundancy criteria.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_models)
            Out-of-fold predictions from base models.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : RedundancyProjector
            Fitted projector.
        """
        X = self._validate_data(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} vs {len(y)}")
        
        self.n_models_initial_ = X.shape[1]
        self.feature_names_in_ = (
            X.columns.tolist() if isinstance(X, pd.DataFrame) 
            else [f"model_{i}" for i in range(X.shape[1])]
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Redundancy Projection (Algorithm 2)")
            print(f"{'='*60}")
            print(f"Initial models: {self.n_models_initial_}")
        
        # Convert to numpy for computation
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Compute initial condition number
        corr_matrix = np.corrcoef(X_array.T)
        self.condition_number_before_ = np.linalg.cond(corr_matrix)
        
        # Step 1: Compute model risks (RMSE)
        model_risks = np.array([
            np.sqrt(mean_squared_error(y, X_array[:, k]))
            for k in range(self.n_models_initial_)
        ])
        
        # Step 2: Sort models by ascending risk (best first)
        sorted_indices = np.argsort(model_risks)
        
        # Step 3: Joint correlation-MSE filtering
        selected_set = []
        self.removed_models_ = []
        
        for idx in sorted_indices:
            retain = True
            
            # Check against all already-selected models
            for selected_idx in selected_set:
                # Compute correlation between predictions
                correlation = np.corrcoef(
                    X_array[:, idx], 
                    X_array[:, selected_idx]
                )[0, 1]
                
                # Compute prediction-space MSE
                pred_mse = mean_squared_error(
                    X_array[:, idx],
                    X_array[:, selected_idx]
                )
                
                # Joint criterion: high correlation AND low MSE
                if correlation >= self.tau_corr and pred_mse <= self.tau_mse:
                    retain = False
                    self.removed_models_.append({
                        'index': idx,
                        'name': self.feature_names_in_[idx],
                        'kept_alternative': self.feature_names_in_[selected_idx],
                        'correlation': correlation,
                        'pred_mse': pred_mse,
                        'rmse': model_risks[idx],
                        'kept_rmse': model_risks[selected_idx],
                        'delta_rmse': model_risks[idx] - model_risks[selected_idx],
                        'reason': 'high_correlation_similar_predictions'
                    })
                    if self.verbose >= 2:
                        print(f"  Removing {self.feature_names_in_[idx]}: "
                              f"ρ={correlation:.3f}, MSE={pred_mse:.4f}, "
                              f"kept {self.feature_names_in_[selected_idx]}")
                    break
            
            if retain:
                selected_set.append(idx)
        
        # Step 4: Variance-based pruning
        if self.tau_var > 0:
            variance_filtered = []
            for idx in selected_set:
                var = np.var(X_array[:, idx])
                if var >= self.tau_var:
                    variance_filtered.append(idx)
                else:
                    self.removed_models_.append({
                        'index': idx,
                        'name': self.feature_names_in_[idx],
                        'variance': var,
                        'reason': 'low_variance'
                    })
                    if self.verbose >= 2:
                        print(f"  Removing {self.feature_names_in_[idx]}: "
                              f"variance={var:.6f} < {self.tau_var}")
            selected_set = variance_filtered
        
        self.selected_indices_ = np.array(sorted(selected_set))
        self.n_models_retained_ = len(self.selected_indices_)
        
        # Compute final condition number
        if self.n_models_retained_ > 1:
            corr_matrix_final = np.corrcoef(X_array[:, self.selected_indices_].T)
            self.condition_number_after_ = np.linalg.cond(corr_matrix_final)
        else:
            self.condition_number_after_ = 1.0
        
        if self.verbose:
            n_removed = self.n_models_initial_ - self.n_models_retained_
            pct_removed = 100 * n_removed / self.n_models_initial_
            cond_improvement = 100 * (1 - self.condition_number_after_ / self.condition_number_before_)
            
            print(f"\nResults:")
            print(f"  Models retained: {self.n_models_retained_}/{self.n_models_initial_} "
                  f"({pct_removed:.1f}% removed)")
            print(f"  Condition number: {self.condition_number_before_:.1f} → "
                  f"{self.condition_number_after_:.1f} "
                  f"({cond_improvement:.1f}% improvement)")
            print(f"  Removed by correlation: {sum(1 for m in self.removed_models_ if m['reason'] == 'high_correlation_similar_predictions')}")
            print(f"  Removed by low variance: {sum(1 for m in self.removed_models_ if m['reason'] == 'low_variance')}")
            print(f"{'='*60}\n")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply projection to select retained models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_models)
            Predictions to project.
        
        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_models_retained)
            Projected predictions containing only retained models.
        """
        if not hasattr(self, 'selected_indices_'):
            raise RuntimeError("Must call fit() before transform()")
        
        X = self._validate_data(X, reset=False)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if X_array.shape[1] != self.n_models_initial_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, expected {self.n_models_initial_}"
            )
        
        return X_array[:, self.selected_indices_]
    
    def get_retained_feature_names(self) -> list:
        """Get names of retained features."""
        if not hasattr(self, 'selected_indices_'):
            raise RuntimeError("Must call fit() before get_retained_feature_names()")
        return [self.feature_names_in_[i] for i in self.selected_indices_]
    
    def get_removal_report(self) -> pd.DataFrame:
        """
        Get detailed report of removed models.
        
        Returns
        -------
        report : pd.DataFrame
            DataFrame with information about each removed model.
        """
        if not hasattr(self, 'removed_models_'):
            raise RuntimeError("Must call fit() before get_removal_report()")
        
        if not self.removed_models_:
            return pd.DataFrame()
        
        return pd.DataFrame(self.removed_models_)
    
    def _validate_data(self, X, reset=True):
        """Validate input data."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)

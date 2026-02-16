"""
Meta-Feature Augmentation

Enriches the prediction space with ensemble statistics and interactions
as described in Section 3.3 of the paper.
"""

import numpy as np
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin


class MetaFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Augments base model predictions with ensemble statistics and interactions.
    
    Implements Phase 2 of the framework: Meta-Feature Augmentation
    
    Features added:
    - μ: Ensemble mean
    - σ: Ensemble standard deviation
    - m: Ensemble median
    - r: Ensemble range (max - min)
    - φ₁: Mean-std interaction (μ × σ)
    - φ₂: Range-std interaction (r × σ)
    
    Parameters
    ----------
    add_statistical : bool, default=True
        Whether to add statistical aggregations (mean, std, median, range).
    
    add_interactions : bool, default=True
        Whether to add interaction features (μσ, rσ).
    
    feature_names : list, optional
        Names for the input features. If None, generates automatic names.
    
    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    
    n_features_out_ : int
        Number of output features after augmentation.
    
    feature_names_out_ : list
        Names of all output features.
    """
    
    def __init__(
        self,
        add_statistical: bool = True,
        add_interactions: bool = True,
        feature_names: list = None
    ):
        self.add_statistical = add_statistical
        self.add_interactions = add_interactions
        self.feature_names = feature_names
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> 'MetaFeatureAugmenter':
        """
        Fit the augmenter (learns feature names).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_models)
            Base model predictions.
        
        y : Ignored
            Not used, present for API consistency.
        
        Returns
        -------
        self : MetaFeatureAugmenter
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
        
        # Determine feature names
        if self.feature_names is not None:
            if len(self.feature_names) != self.n_features_in_:
                raise ValueError(
                    f"feature_names has {len(self.feature_names)} elements, "
                    f"but X has {self.n_features_in_} features"
                )
            self.feature_names_in_ = self.feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"model_{i}" for i in range(self.n_features_in_)]
        
        # Build output feature names
        self.feature_names_out_ = self.feature_names_in_.copy()
        
        if self.add_statistical:
            self.feature_names_out_.extend([
                'ensemble_mean',
                'ensemble_std',
                'ensemble_median',
                'ensemble_range'
            ])
        
        if self.add_interactions:
            self.feature_names_out_.extend([
                'mean_std_interaction',
                'range_std_interaction'
            ])
        
        self.n_features_out_ = len(self.feature_names_out_)
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform predictions by adding meta-features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_models)
            Base model predictions to augment.
        
        Returns
        -------
        X_augmented : ndarray of shape (n_samples, n_features_out)
            Augmented predictions with meta-features.
        """
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError("Must call fit() before transform()")
        
        X = self._validate_data(X, reset=False)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, expected {self.n_features_in_}"
            )
        
        # Start with original features
        features = [X_array]
        
        # Add statistical aggregations
        if self.add_statistical:
            # Mean (μ)
            ensemble_mean = np.mean(X_array, axis=1, keepdims=True)
            features.append(ensemble_mean)
            
            # Standard deviation (σ)
            ensemble_std = np.std(X_array, axis=1, keepdims=True)
            features.append(ensemble_std)
            
            # Median (m)
            ensemble_median = np.median(X_array, axis=1, keepdims=True)
            features.append(ensemble_median)
            
            # Range (r)
            ensemble_max = np.max(X_array, axis=1, keepdims=True)
            ensemble_min = np.min(X_array, axis=1, keepdims=True)
            ensemble_range = ensemble_max - ensemble_min
            features.append(ensemble_range)
            
            # Store for interaction features
            if self.add_interactions:
                mean_vals = ensemble_mean
                std_vals = ensemble_std
                range_vals = ensemble_range
        
        # Add interaction features
        if self.add_interactions:
            if not self.add_statistical:
                # Compute these if not already available
                mean_vals = np.mean(X_array, axis=1, keepdims=True)
                std_vals = np.std(X_array, axis=1, keepdims=True)
                ensemble_max = np.max(X_array, axis=1, keepdims=True)
                ensemble_min = np.min(X_array, axis=1, keepdims=True)
                range_vals = ensemble_max - ensemble_min
            
            # φ₁ = μ × σ
            mean_std_interaction = mean_vals * std_vals
            features.append(mean_std_interaction)
            
            # φ₂ = r × σ
            range_std_interaction = range_vals * std_vals
            features.append(range_std_interaction)
        
        # Concatenate all features
        X_augmented = np.hstack(features)
        
        return X_augmented
    
    def get_feature_names_out(self) -> list:
        """Get output feature names after augmentation."""
        if not hasattr(self, 'feature_names_out_'):
            raise RuntimeError("Must call fit() before get_feature_names_out()")
        return self.feature_names_out_.copy()
    
    def _validate_data(self, X, reset=True):
        """Validate input data."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.asarray(X)

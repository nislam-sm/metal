"""
Meta-Ensemble Blending

Implements Phase 4: Risk-Aware Blending of multiple meta-learners
using inverse-RMSE weighting (Algorithm 4).
"""

import numpy as np
from typing import List
from sklearn.metrics import mean_squared_error


class MetaEnsembleBlender:
    """
    Blends predictions from multiple meta-learners using inverse-RMSE weighting.
    
    Implements Algorithm 4: Inverse-RMSE Meta-Ensemble Blending
    
    This provides a final layer of variance reduction by combining
    Ridge, Lasso, and ElasticNet meta-learners with risk-aware weights.
    
    Parameters
    ----------
    method : str, default='inverse_rmse'
        Blending method. Currently supports:
        - 'inverse_rmse': Weight by 1/RMSE
        - 'uniform': Equal weights
    
    Attributes
    ----------
    weights_ : np.ndarray
        Learned blending weights (sum to 1).
    
    meta_learner_rmses_ : np.ndarray
        Individual RMSEs of each meta-learner.
    
    final_rmse_ : float
        RMSE of the blended predictions.
    
    n_meta_learners_ : int
        Number of meta-learners being blended.
    """
    
    def __init__(self, method: str = 'inverse_rmse'):
        self.method = method
    
    def blend(
        self,
        predictions_list: List[np.ndarray],
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Learn blending weights and return blended predictions.
        
        Parameters
        ----------
        predictions_list : list of arrays
            List of prediction arrays from different meta-learners.
            Each array has shape (n_samples,).
        
        y_true : array-like of shape (n_samples,)
            True target values.
        
        Returns
        -------
        blended_predictions : ndarray of shape (n_samples,)
            Weighted combination of meta-learner predictions.
        """
        if len(predictions_list) == 0:
            raise ValueError("predictions_list cannot be empty")
        
        # Convert to numpy arrays
        predictions_list = [np.asarray(p).ravel() for p in predictions_list]
        y_true = np.asarray(y_true).ravel()
        
        # Validate shapes
        n_samples = len(y_true)
        for i, preds in enumerate(predictions_list):
            if len(preds) != n_samples:
                raise ValueError(
                    f"Prediction {i} has {len(preds)} samples, "
                    f"expected {n_samples}"
                )
        
        self.n_meta_learners_ = len(predictions_list)
        
        # Compute individual RMSEs
        self.meta_learner_rmses_ = np.array([
            np.sqrt(mean_squared_error(y_true, preds))
            for preds in predictions_list
        ])
        
        # Compute weights based on method
        if self.method == 'inverse_rmse':
            # Weight by inverse RMSE: w_m = (1/RMSE_m) / Σ(1/RMSE_m)
            inverse_rmses = 1.0 / self.meta_learner_rmses_
            self.weights_ = inverse_rmses / np.sum(inverse_rmses)
        
        elif self.method == 'uniform':
            # Equal weights
            self.weights_ = np.ones(self.n_meta_learners_) / self.n_meta_learners_
        
        else:
            raise ValueError(f"Unknown blending method: {self.method}")
        
        # Compute blended predictions
        blended = np.zeros(n_samples)
        for weight, preds in zip(self.weights_, predictions_list):
            blended += weight * preds
        
        # Compute final RMSE
        self.final_rmse_ = np.sqrt(mean_squared_error(y_true, blended))
        
        # Store for later use
        self.predictions_list_ = predictions_list
        self.blended_predictions_ = blended
        
        return blended
    
    def predict(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        Apply learned weights to new predictions.
        
        Parameters
        ----------
        predictions_list : list of arrays
            List of prediction arrays from different meta-learners.
        
        Returns
        -------
        blended_predictions : ndarray
            Weighted combination of predictions.
        """
        if not hasattr(self, 'weights_'):
            raise RuntimeError("Must call blend() before predict()")
        
        predictions_list = [np.asarray(p).ravel() for p in predictions_list]
        
        if len(predictions_list) != self.n_meta_learners_:
            raise ValueError(
                f"Expected {self.n_meta_learners_} predictions, "
                f"got {len(predictions_list)}"
            )
        
        n_samples = len(predictions_list[0])
        blended = np.zeros(n_samples)
        
        for weight, preds in zip(self.weights_, predictions_list):
            if len(preds) != n_samples:
                raise ValueError("All predictions must have same length")
            blended += weight * preds
        
        return blended
    
    def get_weights(self) -> dict:
        """
        Get blending weights as a dictionary.
        
        Returns
        -------
        weights_dict : dict
            Mapping from meta-learner index to weight.
        """
        if not hasattr(self, 'weights_'):
            raise RuntimeError("Must call blend() before get_weights()")
        
        return {
            f'meta_learner_{i}': w 
            for i, w in enumerate(self.weights_)
        }
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of the blending.
        
        Returns
        -------
        summary : dict
            Dictionary with blending statistics.
        """
        if not hasattr(self, 'weights_'):
            raise RuntimeError("Must call blend() before get_summary()")
        
        return {
            'n_meta_learners': self.n_meta_learners_,
            'weights': self.weights_.tolist(),
            'individual_rmses': self.meta_learner_rmses_.tolist(),
            'blended_rmse': self.final_rmse_,
            'best_individual_rmse': np.min(self.meta_learner_rmses_),
            'improvement_over_best': (
                np.min(self.meta_learner_rmses_) - self.final_rmse_
            ),
            'improvement_pct': 100 * (
                (np.min(self.meta_learner_rmses_) - self.final_rmse_) /
                np.min(self.meta_learner_rmses_)
            )
        }
    
    def print_summary(self):
        """Print blending summary statistics."""
        if not hasattr(self, 'weights_'):
            raise RuntimeError("Must call blend() before print_summary()")
        
        print(f"\n{'='*60}")
        print("Meta-Ensemble Blending Summary")
        print(f"{'='*60}")
        print(f"Method: {self.method}")
        print(f"Number of meta-learners: {self.n_meta_learners_}")
        print(f"\nIndividual RMSEs:")
        for i, rmse in enumerate(self.meta_learner_rmses_):
            weight = self.weights_[i]
            print(f"  Meta-learner {i}: RMSE={rmse:.6f}, Weight={weight:.4f}")
        
        print(f"\nBlended RMSE: {self.final_rmse_:.6f}")
        best_rmse = np.min(self.meta_learner_rmses_)
        improvement = best_rmse - self.final_rmse_
        improvement_pct = 100 * improvement / best_rmse
        print(f"Best individual: {best_rmse:.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_pct:.3f}%)")
        print(f"{'='*60}\n")

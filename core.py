"""
Regularized Meta-Learning Framework - Core Module

Main interface that combines all four stages of the pipeline.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from sklearn.base import BaseEstimator, RegressorMixin

from .redundancy import RedundancyProjector
from .augmentation import MetaFeatureAugmenter
from .regularizers import RidgeMetaLearner, LassoMetaLearner, ElasticNetMetaLearner
from .blending import MetaEnsembleBlender
from .utils import compute_metrics, validate_predictions, print_metrics


class RegularizedMetaLearner(BaseEstimator, RegressorMixin):
    """
    Complete regularized meta-learning framework.
    
    Implements the full four-stage pipeline:
    1. Redundancy-aware projection (Algorithm 2)
    2. Meta-feature augmentation
    3. Regularized meta-learning (Algorithm 3)
    4. Risk-aware blending (Algorithm 4)
    
    Parameters
    ----------
    tau_corr : float, default=0.95
        Correlation threshold for de-duplication.
    
    tau_mse : float, default=0.01
        MSE threshold for similarity.
    
    tau_var : float, default=0.01
        Variance threshold for low-information models.
    
    add_statistical : bool, default=True
        Whether to add statistical meta-features.
    
    add_interactions : bool, default=True
        Whether to add interaction features.
    
    meta_learners : list, default=['ridge', 'lasso', 'elasticnet']
        Which meta-learners to use. Options: 'ridge', 'lasso', 'elasticnet'.
    
    n_folds : int, default=10
        Number of cross-validation folds.
    
    random_state : int, default=42
        Random seed for reproducibility.
    
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed).
    
    Attributes
    ----------
    projector_ : RedundancyProjector
        Fitted redundancy projector.
    
    augmenter_ : MetaFeatureAugmenter
        Fitted feature augmenter.
    
    meta_learners_ : dict
        Dictionary of fitted meta-learner objects.
    
    blender_ : MetaEnsembleBlender
        Fitted meta-ensemble blender.
    
    oof_predictions_ : np.ndarray
        Final out-of-fold predictions.
    
    oof_rmse_ : float
        Out-of-fold RMSE of final ensemble.
    
    n_models_initial_ : int
        Number of base models before de-duplication.
    
    n_models_retained_ : int
        Number of base models after de-duplication.
    """
    
    def __init__(
        self,
        tau_corr: float = 0.95,
        tau_mse: float = 0.01,
        tau_var: float = 0.01,
        add_statistical: bool = True,
        add_interactions: bool = True,
        meta_learners: List[str] = ['ridge', 'lasso', 'elasticnet'],
        n_folds: int = 10,
        random_state: int = 42,
        verbose: int = 1
    ):
        self.tau_corr = tau_corr
        self.tau_mse = tau_mse
        self.tau_var = tau_var
        self.add_statistical = add_statistical
        self.add_interactions = add_interactions
        self.meta_learners = meta_learners
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> 'RegularizedMetaLearner':
        """
        Fit the complete regularized meta-learning pipeline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_base_models)
            Out-of-fold predictions from base models.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : RegularizedMetaLearner
            Fitted meta-learner.
        """
        # Validate inputs
        validate_predictions(X, y, "Training data")
        
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = np.asarray(y).ravel()
        
        self.n_models_initial_ = X_df.shape[1]
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print("# REGULARIZED META-LEARNING FRAMEWORK")
            print(f"# Paper: https://arxiv.org/abs/2602.12469")
            print(f"{'#'*70}\n")
            print(f"Configuration:")
            print(f"  Initial models: {self.n_models_initial_}")
            print(f"  Meta-learners: {', '.join(self.meta_learners)}")
            print(f"  Cross-validation folds: {self.n_folds}")
            print(f"  Random seed: {self.random_state}")
        
        # ================================================================
        # Phase 1: Redundancy-Aware Projection (Algorithm 2)
        # ================================================================
        if self.verbose:
            print(f"\n{'='*70}")
            print("PHASE 1: Redundancy-Aware Projection")
            print(f"{'='*70}")
        
        self.projector_ = RedundancyProjector(
            tau_corr=self.tau_corr,
            tau_mse=self.tau_mse,
            tau_var=self.tau_var,
            verbose=self.verbose
        )
        
        X_projected = self.projector_.fit_transform(X_df, y)
        self.n_models_retained_ = X_projected.shape[1]
        
        # ================================================================
        # Phase 2: Meta-Feature Augmentation
        # ================================================================
        if self.verbose:
            print(f"\n{'='*70}")
            print("PHASE 2: Meta-Feature Augmentation")
            print(f"{'='*70}")
        
        self.augmenter_ = MetaFeatureAugmenter(
            add_statistical=self.add_statistical,
            add_interactions=self.add_interactions,
            feature_names=self.projector_.get_retained_feature_names()
        )
        
        X_augmented = self.augmenter_.fit_transform(X_projected)
        
        if self.verbose:
            print(f"  Features: {X_projected.shape[1]} → {X_augmented.shape[1]}")
            if self.add_statistical:
                print(f"  Added: mean, std, median, range")
            if self.add_interactions:
                print(f"  Added: μ×σ, r×σ interactions")
        
        # ================================================================
        # Phase 3: Regularized Meta-Learning (Algorithm 3)
        # ================================================================
        if self.verbose:
            print(f"\n{'='*70}")
            print("PHASE 3: Regularized Meta-Learning")
            print(f"{'='*70}")
        
        self.meta_learners_ = {}
        meta_predictions = []
        
        for method in self.meta_learners:
            if self.verbose:
                print(f"\nTraining {method.upper()} meta-learner...")
            
            # Create appropriate meta-learner
            if method == 'ridge':
                learner = RidgeMetaLearner(
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                    verbose=self.verbose
                )
            elif method == 'lasso':
                learner = LassoMetaLearner(
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                    verbose=self.verbose
                )
            elif method == 'elasticnet':
                learner = ElasticNetMetaLearner(
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                    verbose=self.verbose
                )
            else:
                raise ValueError(f"Unknown meta-learner: {method}")
            
            # Fit meta-learner
            learner.fit(X_augmented, y)
            self.meta_learners_[method] = learner
            
            # Get OOF predictions
            meta_predictions.append(learner.predict_oof())
            
            # Print sparsity for Lasso/ElasticNet
            if method in ['lasso', 'elasticnet'] and self.verbose:
                sparsity = learner.get_sparsity()
                n_zero = int(sparsity * X_augmented.shape[1])
                print(f"  Sparsity: {100*sparsity:.1f}% ({n_zero}/{X_augmented.shape[1]} zero weights)")
        
        # ================================================================
        # Phase 4: Meta-Ensemble Blending (Algorithm 4)
        # ================================================================
        if len(self.meta_learners) > 1:
            if self.verbose:
                print(f"\n{'='*70}")
                print("PHASE 4: Meta-Ensemble Blending")
                print(f"{'='*70}")
            
            self.blender_ = MetaEnsembleBlender(method='inverse_rmse')
            self.oof_predictions_ = self.blender_.blend(meta_predictions, y)
            
            if self.verbose:
                self.blender_.print_summary()
        else:
            # Single meta-learner, no blending needed
            self.oof_predictions_ = meta_predictions[0]
            self.blender_ = None
        
        # ================================================================
        # Final Metrics
        # ================================================================
        self.oof_rmse_ = np.sqrt(np.mean((y - self.oof_predictions_) ** 2))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            metrics = compute_metrics(y, self.oof_predictions_)
            print_metrics(metrics, "Out-of-Fold Performance")
            
            # Print comparison
            print("Model Compression:")
            print(f"  Initial models: {self.n_models_initial_}")
            print(f"  Retained models: {self.n_models_retained_}")
            reduction_pct = 100 * (1 - self.n_models_retained_ / self.n_models_initial_)
            print(f"  Reduction: {reduction_pct:.1f}%")
            
            print(f"\nConditioning:")
            print(f"  Before: κ(C) = {self.projector_.condition_number_before_:.1f}")
            print(f"  After:  κ(C) = {self.projector_.condition_number_after_:.1f}")
            improvement = 100 * (1 - self.projector_.condition_number_after_ / 
                               self.projector_.condition_number_before_)
            print(f"  Improvement: {improvement:.1f}%")
            print(f"\n{'='*70}\n")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions on new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_base_models)
            Test predictions from base models.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Meta-learner predictions.
        """
        if not hasattr(self, 'projector_'):
            raise RuntimeError("Must call fit() before predict()")
        
        # Apply pipeline transformations
        X_projected = self.projector_.transform(X)
        X_augmented = self.augmenter_.transform(X_projected)
        
        # Get predictions from each meta-learner
        if len(self.meta_learners_) > 1:
            meta_predictions = [
                learner.predict(X_augmented)
                for learner in self.meta_learners_.values()
            ]
            # Blend predictions
            return self.blender_.predict(meta_predictions)
        else:
            # Single meta-learner
            learner = list(self.meta_learners_.values())[0]
            return learner.predict(X_augmented)
    
    def predict_oof(self) -> np.ndarray:
        """Return out-of-fold predictions from training."""
        if not hasattr(self, 'oof_predictions_'):
            raise RuntimeError("Must call fit() before predict_oof()")
        return self.oof_predictions_.copy()
    
    def evaluate(self, X=None, y=None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Parameters
        ----------
        X : array-like, optional
            Test predictions. If None, uses OOF predictions.
        
        y : array-like, optional
            True targets. Required if X is provided.
        
        Returns
        -------
        metrics : dict
            Dictionary of performance metrics.
        """
        if X is None:
            # Use OOF predictions
            if not hasattr(self, 'oof_predictions_'):
                raise RuntimeError("Must call fit() before evaluate()")
            return compute_metrics(
                self.y_train_ if hasattr(self, 'y_train_') else None,
                self.oof_predictions_
            )
        else:
            # Use provided test data
            if y is None:
                raise ValueError("Must provide y when X is given")
            y_pred = self.predict(X)
            return compute_metrics(y, y_pred)
    
    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get feature importance from Ridge meta-learner.
        
        Parameters
        ----------
        top_k : int, default=10
            Number of top features to return.
        
        Returns
        -------
        importance : pd.DataFrame
            DataFrame with top features and their weights.
        """
        if 'ridge' not in self.meta_learners_:
            raise ValueError("Ridge meta-learner not fitted")
        
        ridge = self.meta_learners_['ridge']
        weights = ridge.get_weights()
        feature_names = self.augmenter_.get_feature_names_out()
        
        # Sort by absolute weight
        abs_weights = np.abs(weights)
        top_indices = np.argsort(abs_weights)[::-1][:top_k]
        
        return pd.DataFrame({
            'rank': range(1, len(top_indices) + 1),
            'feature': [feature_names[i] for i in top_indices],
            'weight': weights[top_indices],
            'abs_weight': abs_weights[top_indices]
        })
    
    def get_removal_report(self) -> pd.DataFrame:
        """Get report of models removed during de-duplication."""
        if not hasattr(self, 'projector_'):
            raise RuntimeError("Must call fit() first")
        return self.projector_.get_removal_report()
    
    def summary(self):
        """Print comprehensive summary of the fitted model."""
        if not hasattr(self, 'projector_'):
            raise RuntimeError("Must call fit() first")
        
        print(f"\n{'='*70}")
        print("REGULARIZED META-LEARNING SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nModel Selection:")
        print(f"  Initial models: {self.n_models_initial_}")
        print(f"  Retained models: {self.n_models_retained_}")
        print(f"  Reduction: {100*(1 - self.n_models_retained_/self.n_models_initial_):.1f}%")
        
        print(f"\nConditioning:")
        print(f"  κ(C) before: {self.projector_.condition_number_before_:.1f}")
        print(f"  κ(C) after: {self.projector_.condition_number_after_:.1f}")
        
        print(f"\nMeta-Learners:")
        for name, learner in self.meta_learners_.items():
            rmse = learner.oof_rmse_
            print(f"  {name.upper()}: RMSE = {rmse:.6f}")
            if hasattr(learner, 'best_lambda_'):
                print(f"           λ = {learner.best_lambda_:.6f}")
        
        if self.blender_ is not None:
            print(f"\nEnsemble Blending:")
            print(f"  Final RMSE: {self.oof_rmse_:.6f}")
            print(f"  Weights: {self.blender_.weights_}")
        
        print(f"{'='*70}\n")

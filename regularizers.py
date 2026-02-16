"""
Regularized Meta-Learners

Implements Phase 3: Ridge, Lasso, and ElasticNet meta-models with
nested cross-validation for hyperparameter selection (Algorithm 3).
"""

import numpy as np
from typing import Union, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings


class BaseMetaLearner(BaseEstimator, RegressorMixin):
    """
    Base class for regularized meta-learners.
    
    Implements Algorithm 3: Nested Cross-Validated Meta-Learning
    """
    
    def __init__(
        self,
        n_folds: int = 10,
        inner_cv: int = 3,
        random_state: Optional[int] = 42,
        verbose: int = 0
    ):
        self.n_folds = n_folds
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMetaLearner':
        """
        Fit meta-learner using nested cross-validation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Meta-features (augmented base predictions).
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : BaseMetaLearner
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError(f"X and y have inconsistent lengths")
        
        self.n_features_in_ = X.shape[1]
        
        # Initialize storage for OOF predictions and fold models
        self.oof_predictions_ = np.zeros(len(y))
        self.fold_models_ = []
        self.fold_scalers_ = []
        self.fold_scores_ = []
        self.fold_lambdas_ = []
        
        # Nested cross-validation
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Standardize features within fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Fit meta-model with inner CV for hyperparameter selection
            model = self._create_model()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
            
            # Make OOF predictions
            val_preds = model.predict(X_val_scaled)
            self.oof_predictions_[val_idx] = val_preds
            
            # Store fold artifacts
            self.fold_models_.append(model)
            self.fold_scalers_.append(scaler)
            
            # Compute and store fold RMSE
            fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            self.fold_scores_.append(fold_rmse)
            
            # Store selected lambda
            if hasattr(model, 'alpha_'):
                self.fold_lambdas_.append(model.alpha_)
            elif hasattr(model, 'lambda_'):
                self.fold_lambdas_.append(model.lambda_)
            
            if self.verbose >= 2:
                lambda_str = f", λ={self.fold_lambdas_[-1]:.6f}" if self.fold_lambdas_ else ""
                print(f"  Fold {fold_idx + 1}/{self.n_folds}: "
                      f"RMSE={fold_rmse:.6f}{lambda_str}")
        
        # Compute overall OOF RMSE
        self.oof_rmse_ = np.sqrt(mean_squared_error(y, self.oof_predictions_))
        self.oof_score_ = self.oof_rmse_  # For compatibility
        
        # Store mean and std of selected hyperparameters
        if self.fold_lambdas_:
            self.best_lambda_ = np.mean(self.fold_lambdas_)
            self.lambda_std_ = np.std(self.fold_lambdas_)
        
        # Fit final model on all data for prediction
        self.final_scaler_ = StandardScaler()
        X_scaled = self.final_scaler_.fit_transform(X)
        self.final_model_ = self._create_model()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.final_model_.fit(X_scaled, y)
        
        if self.verbose:
            print(f"  OOF RMSE: {self.oof_rmse_:.6f} ± {np.std(self.fold_scores_):.6f}")
            if hasattr(self, 'best_lambda_'):
                print(f"  Mean λ: {self.best_lambda_:.6f} ± {self.lambda_std_:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the final model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("Must call fit() before predict()")
        
        X = np.asarray(X)
        X_scaled = self.final_scaler_.transform(X)
        return self.final_model_.predict(X_scaled)
    
    def predict_oof(self) -> np.ndarray:
        """Return out-of-fold predictions from training."""
        if not hasattr(self, 'oof_predictions_'):
            raise RuntimeError("Must call fit() before predict_oof()")
        return self.oof_predictions_.copy()
    
    def get_weights(self) -> np.ndarray:
        """Get learned meta-model weights."""
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("Must call fit() before get_weights()")
        return self.final_model_.coef_
    
    def _create_model(self):
        """Create the underlying sklearn model. Override in subclasses."""
        raise NotImplementedError


class RidgeMetaLearner(BaseMetaLearner):
    """
    Ridge regression meta-learner with nested CV.
    
    Implements L2 regularization: Ω(w) = λ‖w‖₂²
    
    Parameters
    ----------
    alphas : array-like, optional
        Grid of alpha values for cross-validation.
        Default: np.logspace(-3, 5, 50)
    
    n_folds : int, default=10
        Number of outer cross-validation folds.
    
    inner_cv : int, default=3
        Number of inner cross-validation folds for hyperparameter selection.
    
    random_state : int, default=42
        Random seed for reproducibility.
    
    verbose : int, default=0
        Verbosity level.
    """
    
    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        n_folds: int = 10,
        inner_cv: int = 3,
        random_state: Optional[int] = 42,
        verbose: int = 0
    ):
        super().__init__(n_folds, inner_cv, random_state, verbose)
        self.alphas = alphas if alphas is not None else np.logspace(-3, 5, 50)
    
    def _create_model(self):
        return RidgeCV(
            alphas=self.alphas,
            cv=self.inner_cv,
            scoring='neg_root_mean_squared_error'
        )


class LassoMetaLearner(BaseMetaLearner):
    """
    Lasso regression meta-learner with nested CV.
    
    Implements L1 regularization: Ω(w) = λ‖w‖₁
    Induces sparsity (some weights driven to exactly zero).
    
    Parameters
    ----------
    alphas : array-like, optional
        Grid of alpha values for cross-validation.
        Default: np.logspace(-5, 1, 30)
    
    n_folds : int, default=10
        Number of outer cross-validation folds.
    
    inner_cv : int, default=3
        Number of inner cross-validation folds.
    
    random_state : int, default=42
        Random seed.
    
    verbose : int, default=0
        Verbosity level.
    """
    
    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        n_folds: int = 10,
        inner_cv: int = 3,
        random_state: Optional[int] = 42,
        verbose: int = 0
    ):
        super().__init__(n_folds, inner_cv, random_state, verbose)
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
    
    def _create_model(self):
        return LassoCV(
            alphas=self.alphas,
            cv=self.inner_cv,
            random_state=self.random_state,
            max_iter=10000
        )
    
    def get_sparsity(self) -> float:
        """
        Get fraction of zero weights (sparsity level).
        
        Returns
        -------
        sparsity : float
            Fraction of weights that are exactly zero.
        """
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("Must call fit() before get_sparsity()")
        
        weights = self.final_model_.coef_
        n_zero = np.sum(np.abs(weights) < 1e-10)
        return n_zero / len(weights)


class ElasticNetMetaLearner(BaseMetaLearner):
    """
    ElasticNet regression meta-learner with nested CV.
    
    Implements mixed L1/L2 regularization: Ω(w) = λ₁‖w‖₁ + λ₂‖w‖₂²
    Balances Ridge stability with Lasso sparsity.
    
    Parameters
    ----------
    alphas : array-like, optional
        Grid of alpha values (overall regularization strength).
        Default: np.logspace(-5, 1, 30)
    
    l1_ratio : array-like, default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        Grid of L1/L2 mixing parameters.
        l1_ratio=1 is pure Lasso, l1_ratio=0 is pure Ridge.
    
    n_folds : int, default=10
        Number of outer cross-validation folds.
    
    inner_cv : int, default=3
        Number of inner cross-validation folds.
    
    random_state : int, default=42
        Random seed.
    
    verbose : int, default=0
        Verbosity level.
    """
    
    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        l1_ratio: list = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        n_folds: int = 10,
        inner_cv: int = 3,
        random_state: Optional[int] = 42,
        verbose: int = 0
    ):
        super().__init__(n_folds, inner_cv, random_state, verbose)
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
        self.l1_ratio = l1_ratio
    
    def _create_model(self):
        return ElasticNetCV(
            alphas=self.alphas,
            l1_ratio=self.l1_ratio,
            cv=self.inner_cv,
            random_state=self.random_state,
            max_iter=10000
        )
    
    def get_sparsity(self) -> float:
        """Get fraction of zero weights."""
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("Must call fit() before get_sparsity()")
        
        weights = self.final_model_.coef_
        n_zero = np.sum(np.abs(weights) < 1e-10)
        return n_zero / len(weights)
    
    def get_l1_ratio(self) -> float:
        """Get selected L1/L2 mixing ratio."""
        if not hasattr(self, 'final_model_'):
            raise RuntimeError("Must call fit() before get_l1_ratio()")
        
        return self.final_model_.l1_ratio_

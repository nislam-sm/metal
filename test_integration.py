"""
Simple integration test for the regularized meta-learning framework.

Run with: pytest tests/test_integration.py -v
"""

import numpy as np
import pytest
from meta_ensemble import RegularizedMetaLearner
from meta_ensemble.utils import compute_metrics


def generate_synthetic_data(n_samples=1000, n_models=20, random_state=42):
    """Generate synthetic base model predictions for testing."""
    np.random.seed(random_state)
    
    # True function: y = 3*x1 + 2*x2 + noise
    X = np.random.randn(n_samples, 2)
    y_true = 3 * X[:, 0] + 2 * X[:, 1] + 0.5 * np.random.randn(n_samples)
    
    # Generate diverse base model predictions
    base_predictions = []
    for i in range(n_models):
        # Each model has different weights + noise
        w1 = 2.5 + 0.5 * np.random.randn()
        w2 = 1.5 + 0.5 * np.random.randn()
        noise_level = 0.3 + 0.2 * np.random.rand()
        
        y_pred = w1 * X[:, 0] + w2 * X[:, 1] + noise_level * np.random.randn(n_samples)
        base_predictions.append(y_pred)
    
    predictions = np.column_stack(base_predictions)
    
    return predictions, y_true


class TestRegularizedMetaLearner:
    """Test suite for RegularizedMetaLearner."""
    
    def test_basic_fit_predict(self):
        """Test basic fit and predict workflow."""
        # Generate data
        X_train, y_train = generate_synthetic_data(n_samples=500, n_models=15)
        X_test, y_test = generate_synthetic_data(n_samples=200, n_models=15, random_state=43)
        
        # Initialize and fit
        model = RegularizedMetaLearner(
            n_folds=3,  # Small for speed
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # Check attributes
        assert hasattr(model, 'projector_')
        assert hasattr(model, 'augmenter_')
        assert hasattr(model, 'meta_learners_')
        assert hasattr(model, 'oof_predictions_')
        
        # Check OOF predictions
        oof_preds = model.predict_oof()
        assert len(oof_preds) == len(y_train)
        assert not np.any(np.isnan(oof_preds))
        
        # Check test predictions
        test_preds = model.predict(X_test)
        assert len(test_preds) == len(y_test)
        assert not np.any(np.isnan(test_preds))
    
    def test_redundancy_projection(self):
        """Test that redundancy projection reduces model count."""
        # Generate data with duplicate models
        X_train, y_train = generate_synthetic_data(n_samples=500, n_models=20)
        
        # Add duplicate models (exactly the same)
        X_train_dup = np.column_stack([
            X_train,
            X_train[:, 0],  # Duplicate of first model
            X_train[:, 1],  # Duplicate of second model
        ])
        
        model = RegularizedMetaLearner(
            tau_corr=0.95,
            n_folds=3,
            random_state=42,
            verbose=0
        )
        model.fit(X_train_dup, y_train)
        
        # Should remove at least the duplicates
        assert model.n_models_retained_ < model.n_models_initial_
    
    def test_meta_feature_augmentation(self):
        """Test that meta-features are added correctly."""
        X_train, y_train = generate_synthetic_data(n_samples=500, n_models=10)
        
        model = RegularizedMetaLearner(
            add_statistical=True,
            add_interactions=True,
            n_folds=3,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # Check that augmented features were added
        feature_names = model.augmenter_.get_feature_names_out()
        assert 'ensemble_mean' in feature_names
        assert 'ensemble_std' in feature_names
        assert 'mean_std_interaction' in feature_names
    
    def test_performance_improvement(self):
        """Test that framework improves over simple averaging."""
        X_train, y_train = generate_synthetic_data(n_samples=1000, n_models=20)
        
        # Simple average baseline
        simple_avg = np.mean(X_train, axis=1)
        simple_rmse = np.sqrt(np.mean((y_train - simple_avg) ** 2))
        
        # Our method
        model = RegularizedMetaLearner(
            n_folds=5,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        our_rmse = model.oof_rmse_
        
        # Should improve over simple averaging
        assert our_rmse < simple_rmse, \
            f"Our RMSE ({our_rmse:.4f}) should be better than simple average ({simple_rmse:.4f})"
    
    def test_single_meta_learner(self):
        """Test with single meta-learner (no blending)."""
        X_train, y_train = generate_synthetic_data(n_samples=500, n_models=15)
        
        model = RegularizedMetaLearner(
            meta_learners=['ridge'],  # Only one
            n_folds=3,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # Should not have blender
        assert model.blender_ is None
        
        # Should still make predictions
        oof_preds = model.predict_oof()
        assert len(oof_preds) == len(y_train)
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        metrics = compute_metrics(y_true, y_pred)
        
        # Check all metrics present
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'pearson' in metrics
        
        # Check reasonable values
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2'] <= 1
        assert -1 <= metrics['pearson'] <= 1
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X_train, y_train = generate_synthetic_data(n_samples=500, n_models=15)
        
        model = RegularizedMetaLearner(
            n_folds=3,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # Get feature importance
        importance = model.get_feature_importance(top_k=10)
        
        # Check structure
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'weight' in importance.columns
        assert len(importance) == min(10, model.augmenter_.n_features_out_)


def test_quick_smoke():
    """Quick smoke test that everything imports and runs."""
    X, y = generate_synthetic_data(n_samples=100, n_models=5)
    
    model = RegularizedMetaLearner(
        n_folds=2,
        meta_learners=['ridge'],
        random_state=42,
        verbose=0
    )
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == len(y)
    print("✓ Smoke test passed!")


if __name__ == '__main__':
    # Run smoke test
    test_quick_smoke()
    
    # Run all tests
    pytest.main([__file__, '-v'])

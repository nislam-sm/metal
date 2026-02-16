"""
Playground Series S6E1 Benchmark Experiment

Reproduces the main results from Table 1 of the paper.

Usage:
    python playground_s6e1.py --data-path /path/to/data --output-dir results/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_ensemble import RegularizedMetaLearner
from meta_ensemble.utils import compute_metrics, print_metrics


def load_base_model_predictions(data_path: str):
    """
    Load OOF and test predictions from base models.
    
    Expected structure:
    data_path/
        oof_predictions.csv    # (N_train, K_models)
        test_predictions.csv   # (N_test, K_models)
        train.csv              # (N_train, features + target)
    
    Returns
    -------
    oof_preds : pd.DataFrame
        Out-of-fold predictions from base models.
    
    test_preds : pd.DataFrame
        Test predictions from base models.
    
    y_train : np.ndarray
        Training targets.
    """
    data_path = Path(data_path)
    
    print(f"Loading data from {data_path}...")
    
    # Load OOF predictions
    oof_preds = pd.read_csv(data_path / "oof_predictions.csv")
    print(f"  OOF predictions: {oof_preds.shape}")
    
    # Load test predictions
    test_preds = pd.read_csv(data_path / "test_predictions.csv")
    print(f"  Test predictions: {test_preds.shape}")
    
    # Load training data for targets
    train_df = pd.read_csv(data_path / "train.csv")
    y_train = train_df['exam_score'].values
    print(f"  Training targets: {len(y_train)}")
    
    return oof_preds, test_preds, y_train


def run_baselines(oof_preds: pd.DataFrame, y_train: np.ndarray):
    """
    Compute baseline methods for comparison.
    
    Returns
    -------
    baseline_results : dict
        Dictionary of baseline predictions and metrics.
    """
    print(f"\n{'='*70}")
    print("COMPUTING BASELINES")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Best Single Model
    print("Best Single Model...")
    single_rmses = []
    for col in oof_preds.columns:
        rmse = np.sqrt(np.mean((y_train - oof_preds[col].values) ** 2))
        single_rmses.append((col, rmse))
    
    best_model, best_rmse = min(single_rmses, key=lambda x: x[1])
    results['best_single'] = {
        'predictions': oof_preds[best_model].values,
        'metrics': compute_metrics(y_train, oof_preds[best_model].values),
        'model': best_model
    }
    print(f"  Model: {best_model}, RMSE: {best_rmse:.6f}")
    
    # Simple Average
    print("\nSimple Average...")
    simple_avg = oof_preds.mean(axis=1).values
    results['simple_average'] = {
        'predictions': simple_avg,
        'metrics': compute_metrics(y_train, simple_avg)
    }
    print(f"  RMSE: {results['simple_average']['metrics']['rmse']:.6f}")
    
    # Weighted Average (Performance-based)
    print("\nWeighted Average (Performance)...")
    rmses = np.array([rmse for _, rmse in single_rmses])
    weights = (1 / rmses) / np.sum(1 / rmses)
    weighted_avg = np.average(oof_preds.values, axis=1, weights=weights)
    results['weighted_average'] = {
        'predictions': weighted_avg,
        'metrics': compute_metrics(y_train, weighted_avg)
    }
    print(f"  RMSE: {results['weighted_average']['metrics']['rmse']:.6f}")
    
    # Vanilla Ridge Stacking
    print("\nVanilla Ridge Stacking...")
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 50), cv=5)
    vanilla_ridge = cross_val_predict(ridge, oof_preds.values, y_train, cv=10)
    results['vanilla_ridge'] = {
        'predictions': vanilla_ridge,
        'metrics': compute_metrics(y_train, vanilla_ridge)
    }
    print(f"  RMSE: {results['vanilla_ridge']['metrics']['rmse']:.6f}")
    
    return results


def run_regularized_metalearning(
    oof_preds: pd.DataFrame,
    y_train: np.ndarray,
    config: dict
):
    """
    Run the full regularized meta-learning framework.
    
    Parameters
    ----------
    oof_preds : pd.DataFrame
        Out-of-fold predictions.
    
    y_train : np.ndarray
        Training targets.
    
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    model : RegularizedMetaLearner
        Fitted meta-learner.
    
    results : dict
        Results dictionary.
    """
    print(f"\n{'='*70}")
    print("REGULARIZED META-LEARNING FRAMEWORK")
    print(f"{'='*70}\n")
    
    # Initialize model
    model = RegularizedMetaLearner(
        tau_corr=config['tau_corr'],
        tau_mse=config['tau_mse'],
        tau_var=config['tau_var'],
        add_statistical=config['add_statistical'],
        add_interactions=config['add_interactions'],
        meta_learners=config['meta_learners'],
        n_folds=config['n_folds'],
        random_state=config['random_state'],
        verbose=config['verbose']
    )
    
    # Fit model
    start_time = time.time()
    model.fit(oof_preds, y_train)
    training_time = time.time() - start_time
    
    # Get results
    results = {
        'model': model,
        'predictions': model.predict_oof(),
        'metrics': model.evaluate(),
        'training_time': training_time,
    }
    
    print(f"\nTraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    return model, results


def compare_results(baseline_results: dict, rml_results: dict):
    """Print comparison table of all methods."""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    # Create comparison table
    comparison = []
    
    # Baselines
    comparison.append({
        'Method': 'Best Single Model',
        'RMSE': baseline_results['best_single']['metrics']['rmse'],
        'MAE': baseline_results['best_single']['metrics']['mae'],
        'R²': baseline_results['best_single']['metrics']['r2'],
        'Models': 1
    })
    
    comparison.append({
        'Method': 'Simple Average',
        'RMSE': baseline_results['simple_average']['metrics']['rmse'],
        'MAE': baseline_results['simple_average']['metrics']['mae'],
        'R²': baseline_results['simple_average']['metrics']['r2'],
        'Models': len(baseline_results['simple_average']['predictions'])
    })
    
    comparison.append({
        'Method': 'Weighted Average',
        'RMSE': baseline_results['weighted_average']['metrics']['rmse'],
        'MAE': baseline_results['weighted_average']['metrics']['mae'],
        'R²': baseline_results['weighted_average']['metrics']['r2'],
        'Models': len(baseline_results['weighted_average']['predictions'])
    })
    
    comparison.append({
        'Method': 'Vanilla Ridge Stack',
        'RMSE': baseline_results['vanilla_ridge']['metrics']['rmse'],
        'MAE': baseline_results['vanilla_ridge']['metrics']['mae'],
        'R²': baseline_results['vanilla_ridge']['metrics']['r2'],
        'Models': '-'
    })
    
    # Our method
    comparison.append({
        'Method': 'Ours (Full Ensemble)',
        'RMSE': rml_results['metrics']['rmse'],
        'MAE': rml_results['metrics']['mae'],
        'R²': rml_results['metrics']['r2'],
        'Models': rml_results['model'].n_models_retained_
    })
    
    # Print table
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    # Print improvements
    print(f"\n{'='*70}")
    print("RELATIVE IMPROVEMENTS")
    print(f"{'='*70}\n")
    
    our_rmse = rml_results['metrics']['rmse']
    
    improvements = {
        'vs. Best Single': (baseline_results['best_single']['metrics']['rmse'] - our_rmse) / baseline_results['best_single']['metrics']['rmse'] * 100,
        'vs. Simple Average': (baseline_results['simple_average']['metrics']['rmse'] - our_rmse) / baseline_results['simple_average']['metrics']['rmse'] * 100,
        'vs. Vanilla Ridge': (baseline_results['vanilla_ridge']['metrics']['rmse'] - our_rmse) / baseline_results['vanilla_ridge']['metrics']['rmse'] * 100,
    }
    
    for method, improvement in improvements.items():
        print(f"{method:25s}: {improvement:+.2f}%")


def save_results(results: dict, output_dir: str):
    """Save results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save predictions
    pd.DataFrame({
        'oof_predictions': results['predictions']
    }).to_csv(output_dir / 'oof_predictions.csv', index=False)
    
    # Save metrics
    pd.DataFrame([results['metrics']]).to_csv(
        output_dir / 'metrics.csv', index=False
    )
    
    # Save removal report
    removal_report = results['model'].get_removal_report()
    if not removal_report.empty:
        removal_report.to_csv(output_dir / 'removed_models.csv', index=False)
    
    # Save feature importance
    feature_importance = results['model'].get_feature_importance(top_k=20)
    feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print("Results saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run Playground S6E1 benchmark experiment"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=10,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0=silent, 1=progress, 2=detailed)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'tau_corr': 0.95,
        'tau_mse': 0.01,
        'tau_var': 0.01,
        'add_statistical': True,
        'add_interactions': True,
        'meta_learners': ['ridge', 'lasso', 'elasticnet'],
        'n_folds': args.n_folds,
        'random_state': args.seed,
        'verbose': args.verbose
    }
    
    # Load data
    oof_preds, test_preds, y_train = load_base_model_predictions(args.data_path)
    
    # Run baselines
    baseline_results = run_baselines(oof_preds, y_train)
    
    # Run our method
    model, rml_results = run_regularized_metalearning(oof_preds, y_train, config)
    
    # Compare results
    compare_results(baseline_results, rml_results)
    
    # Save results
    save_results(rml_results, args.output_dir)
    
    # Generate test predictions
    print(f"\nGenerating test predictions...")
    test_predictions = model.predict(test_preds)
    
    pd.DataFrame({
        'id': range(len(test_predictions)),
        'exam_score': test_predictions
    }).to_csv(Path(args.output_dir) / 'submission.csv', index=False)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {args.output_dir}")
    print(f"Test predictions: {args.output_dir}/submission.csv")


if __name__ == '__main__':
    main()

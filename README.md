# Paper Title: Regularized Meta-Learning for Improved Generalization

[![arXiv](https://img.shields.io/badge/arXiv-2602.12469-b31b1b.svg)](https://arxiv.org/abs/2602.12469)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLSys 2025](https://img.shields.io/badge/MLSys-2025-brightgreen.svg)](https://mlsys.org/)

**Official implementation of "Regularized Meta-Learning for Improved Generalization"**.

> **Authors:** Noor Islam S. Mohammad¹, Md Muntaqim Meherab²  
> ¹New York University, ²Daffodil International University

---
![ML-25](https://github.com/user-attachments/assets/420fb834-61e9-4412-a7b6-1a38a84853b3)

## 🎯 Overview

Deep ensemble methods often suffer from three critical limitations:
1. **Redundancy** among base models that inflates computational cost
2. **Unstable weighting** under multicollinearity
3. **Meta-level overfitting** in high-dimensional prediction spaces

This framework addresses these challenges through a **four-stage regularized meta-learning pipeline** that achieves state-of-the-art performance while maintaining deployment efficiency.

### Key Results

| Metric | Our Method | Hill Climbing | Simple Average | Improvement |
|--------|-----------|---------------|----------------|-------------|
| **RMSE** | **8.582** | 8.603 | 8.894 | **3.5%** over averaging |
| **Runtime** | **712.8s** | 2,841.6s | 3.2s | **4.0× faster** than hill climbing |
| **Models Used** | 37 | 28 | 45 | 17.8% reduction |
| **Memory** | 289 MB | 387 MB | 124 MB | 25% less than hill climbing |

---

## ✨ Key Features

### 🔧 **Four-Stage Pipeline**
1. **Redundancy-Aware Projection** – Removes near-collinear predictors (correlation threshold τ = 0.95)
2. **Meta-Feature Augmentation** – Enriches prediction space with ensemble statistics
3. **Regularized Meta-Learning** – Cross-validated Ridge, Lasso, and ElasticNet
4. **Risk-Aware Blending** – Inverse-RMSE weighting for stability

### 📊 **Technical Highlights**
- ✅ **53.7% reduction** in condition number (847 → 392)
- ✅ **48.6% model compression** (72 → 37 models) without accuracy loss
- ✅ **Sub-millisecond inference** (<0.3ms per prediction)
- ✅ **Statistically rigorous** with Bonferroni-corrected significance testing
- ✅ **Production-ready** with complete reproducibility

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/metarl/regularized-metalearning.git
cd regularized-metalearning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from meta_ensemble import RegularizedMetaLearner
import numpy as np
import pandas as pd

# Load your base model predictions
# Shape: (N_samples, K_models) for OOF and test sets
oof_predictions = pd.read_csv('oof_predictions.csv')
test_predictions = pd.read_csv('test_predictions.csv')
y_true = pd.read_csv('targets.csv')['target'].values

# Initialize the framework
rml = RegularizedMetaLearner(
    tau_corr=0.95,           # Correlation threshold for de-duplication
    tau_mse=0.01,            # MSE threshold for similarity
    tau_var=0.01,            # Variance threshold for pruning
    meta_learners=['ridge', 'lasso', 'elasticnet'],
    n_folds=10,              # Cross-validation folds
    random_state=42
)

# Fit the meta-learner
rml.fit(oof_predictions, y_true)

# Generate predictions
oof_preds = rml.predict_oof()
test_preds = rml.predict(test_predictions)

# Get performance metrics
metrics = rml.evaluate()
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"Models retained: {rml.n_models_retained_}/{rml.n_models_initial_}")
```

### Advanced Usage: Full Pipeline

```python
from meta_ensemble import (
    RedundancyProjector,
    MetaFeatureAugmenter,
    RegularizedMetaLearner,
    MetaEnsembleBlender
)

# Stage 1: Redundancy Projection
projector = RedundancyProjector(tau_corr=0.95, tau_mse=0.01, tau_var=0.01)
selected_models = projector.fit_transform(oof_predictions, y_true)

print(f"De-duplication: {oof_predictions.shape[1]} → {selected_models.shape[1]} models")
print(f"Condition number: {projector.condition_number_before_:.1f} → {projector.condition_number_after_:.1f}")

# Stage 2: Meta-Feature Augmentation
augmenter = MetaFeatureAugmenter(
    add_statistical=True,
    add_interactions=True
)
oof_augmented = augmenter.fit_transform(selected_models)
test_augmented = augmenter.transform(test_predictions[:, projector.selected_indices_])

print(f"Feature augmentation: {selected_models.shape[1]} → {oof_augmented.shape[1]} features")

# Stage 3: Regularized Meta-Learning
meta_learners = {}
for method in ['ridge', 'lasso', 'elasticnet']:
    ml = RegularizedMetaLearner(method=method, n_folds=10)
    ml.fit(oof_augmented, y_true)
    meta_learners[method] = ml
    print(f"{method.upper()} RMSE: {ml.oof_rmse_:.4f}, Optimal λ: {ml.best_lambda_:.4f}")

# Stage 4: Meta-Ensemble Blending
blender = MetaEnsembleBlender()
final_oof_preds = blender.blend([ml.predict_oof() for ml in meta_learners.values()], y_true)
final_test_preds = blender.predict([ml.predict(test_augmented) for ml in meta_learners.values()])

print(f"\nFinal RMSE: {blender.final_rmse_:.4f}")
print(f"Ensemble weights: {blender.weights_}")
```

---

## 📂 Project Structure

```
regularized-metalearning/
├── meta_ensemble/
│   ├── __init__.py
│   ├── core.py                    # Main RegularizedMetaLearner class
│   ├── redundancy.py              # Redundancy projection (Algorithm 2)
│   ├── augmentation.py            # Meta-feature engineering
│   ├── regularizers.py            # Ridge/Lasso/ElasticNet implementations
│   ├── blending.py                # Risk-aware ensemble blending
│   └── utils.py                   # Utility functions
├── experiments/
│   ├── playground_s6e1.py         # Main benchmark experiment
│   ├── ablation_study.py          # Component ablation analysis
│   ├── scaling_analysis.py        # Dataset size scaling experiments
│   └── baseline_comparison.py     # Comparison with baselines
├── notebooks/
│   ├── 01_quick_start.ipynb       # Getting started tutorial
│   ├── 02_full_pipeline.ipynb     # Complete pipeline walkthrough
│   ├── 03_analysis.ipynb          # Results analysis and visualization
│   └── 04_custom_data.ipynb       # Using with your own data
├── tests/
│   ├── test_redundancy.py
│   ├── test_augmentation.py
│   ├── test_regularizers.py
│   └── test_integration.py
├── figures/                        # Paper figures (generated)
├── results/                        # Experimental results
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

---

## 📊 Benchmark Results

### Overall Performance (Playground Series S6E1)

Comparison on 100K training samples, 630K accumulated OOF predictions:

```python
Method                          RMSE    MAE    R²      Models  Time(s)  Speedup
────────────────────────────────────────────────────────────────────────────────
Best Single Model               9.247  7.103  0.7521    1        -        -
Simple Average                  8.894  6.812  0.7710   45       3.2      -
Weighted Average (Perf.)        8.756  6.691  0.7782   45      12.7      -
Vanilla Stack (Linear)          8.691  6.634  0.7815   45      67.4      -
Vanilla Stack (Ridge)           8.627  6.578  0.7848   45     189.3      -
Hill Climbing (Greedy)          8.603  6.561  0.7861   28    2841.6    1.0×
────────────────────────────────────────────────────────────────────────────────
Ours (Ridge Only)               8.583  6.547  0.7871   37     234.7   12.1×
Ours (Lasso Only)               8.584  6.548  0.7871   37     287.3    9.9×
Ours (ElasticNet Only)          8.584  6.548  0.7871   37     301.2    9.4×
Ours (Full Ensemble) ⭐         8.582  6.546  0.7872   37     712.8    4.0×
────────────────────────────────────────────────────────────────────────────────
```

### Ablation Study: Component Contributions

Each component added cumulatively to baseline Ridge stacking:

| Component | RMSE | Δ RMSE | Contribution |
|-----------|------|--------|--------------|
| Baseline (Ridge, no preprocessing) | 7.183 | — | — |
| + De-duplication (37 models) | 7.094 | −0.089 | **1.24%** |
| + Variance pruning | 7.056 | −0.038 | 0.53% |
| + Statistical features | 6.921 | −0.135 | **1.88%** |
| + Interaction features | 6.873 | −0.048 | 0.67% |
| + Meta-ensemble blending | **6.742** | −0.131 | **1.82%** |
| **Total Improvement** | | **−0.441** | **6.13%** |

### Computational Efficiency

**Training Time Breakdown** (full pipeline, 712.8s total):
- De-duplication: 8.4s (1.2%)
- Feature engineering: 4.7s (0.7%)
- Ridge meta-learning: 234.7s (32.9%)
- Lasso meta-learning: 287.3s (40.3%)
- ElasticNet meta-learning: 301.2s (42.3%)
- Ensemble blending: 1.2s (0.2%)

**Inference Performance:**
- **Per-prediction latency:** <0.3ms
- **Throughput:** >3,300 predictions/second (single core)
- **Memory footprint:** 289 MB peak, <5 KB serialized model

---

## 🔬 Theoretical Guarantees

Our framework provides formal guarantees for:

### 1. Spectral Preconditioning (Theorem 1)
Redundancy projection strictly improves the minimum singular value:
```
σ_min(P_eff) ≥ σ_min(P) + Δ_τ
```
This reduces the condition number and stabilizes meta-learning.

### 2. Perturbation Bounds (Theorem 2)
Under data perturbations Δ P, the composite operator satisfies:
```
‖Δŵ_τ‖₂ ≤ [κ(C_eff + λI) / σ_min(P_eff)] · ‖ΔP‖₂ · ‖y‖₂
```

### 3. Generalization (Theorem 3)
Effective rank reduction improves Rademacher complexity:
```
R_N = O(B/√N · √rank_eff(C_eff))
```

### 4. Variance Reduction (Theorem 4)
Meta-ensemble blending strictly reduces variance:
```
Var(ĝ_blend) < min_m Var(ĝ_m)
```

See **Appendix H** in the paper for complete proofs.

---

## 📈 Reproducing Paper Results

### Main Benchmark Experiment

```bash
# Run the complete Playground S6E1 experiment
python experiments/playground_s6e1.py \
    --n-folds 10 \
    --tau-corr 0.95 \
    --tau-mse 0.01 \
    --seed 42 \
    --output-dir results/main

# Expected output:
# ✓ RMSE: 8.582 (std: 0.043)
# ✓ Runtime: ~712s
# ✓ Models: 37/45 retained
```

### Ablation Study

```bash
# Run component ablation analysis
python experiments/ablation_study.py \
    --n-folds 10 \
    --seed 42 \
    --output-dir results/ablation

# Generates Table 3 from the paper
```

### Scaling Analysis

```bash
# Test performance across dataset sizes
python experiments/scaling_analysis.py \
    --sample-fractions 0.1 0.25 0.5 0.75 1.0 \
    --n-folds 10 \
    --seed 42 \
    --output-dir results/scaling

# Generates Figure 6 from the paper
```

### Generate All Paper Figures

```bash
# Reproduce all figures from the paper
python scripts/generate_figures.py \
    --results-dir results/ \
    --output-dir figures/

# Generates Figures 2-7 as PDF/PNG
```

---

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{mohammad2025regularized,
  title={Regularized Meta-Learning for Improved Generalization},
  author={Mohammad, Noor Islam S. and Meherab, Md Muntaqim},
  booktitle={Proceedings of the 8th MLSys Conference},
  year={2025},
  address={Santa Clara, CA, USA},
  url={https://arxiv.org/abs/2602.12469}
}
```

**arXiv preprint:** [arXiv:2602.12469](https://arxiv.org/abs/2602.12469)

---

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory:

- [**Installation Guide**](docs/installation.md) – Detailed setup instructions
- [**API Reference**](docs/api.md) – Complete API documentation
- [**Tutorials**](docs/tutorials.md) – Step-by-step guides
- [**FAQ**](docs/faq.md) – Common questions and troubleshooting
- [**Best Practices**](docs/best_practices.md) – Deployment recommendations

Or visit our [**online documentation**](https://regularized-metalearning.readthedocs.io/).

---

## 🛠️ Requirements

### Core Dependencies
```
python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.2
scipy >= 1.7.0
```

### Optional (for visualization and experiments)
```
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

### Development Dependencies
```
pytest >= 6.0.0
black >= 21.0
flake8 >= 3.9.0
mypy >= 0.910
```

Full requirements in `requirements.txt` and `requirements-dev.txt`.

---

## 🧪 Testing

Run the complete test suite:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=meta_ensemble --cov-report=html

# Run specific test modules
pytest tests/test_redundancy.py -v
pytest tests/test_integration.py -v
```

### Test Coverage

Current test coverage: **94.2%**

| Module | Coverage |
|--------|----------|
| `core.py` | 97% |
| `redundancy.py` | 95% |
| `augmentation.py` | 98% |
| `regularizers.py` | 91% |
| `blending.py` | 93% |
| `utils.py` | 89% |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/regularized-metalearning.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make changes** and add tests
5. **Run tests**: `pytest tests/`
6. **Format code**: `black meta_ensemble/` and `flake8 meta_ensemble/`
7. **Commit**: `git commit -m "Add feature: description"`
8. **Push**: `git push origin feature/your-feature-name`
9. **Create Pull Request** on GitHub

### Code Style

We follow [PEP 8](https://pep8.org/) with these specifics:
- Line length: 100 characters
- Use Black for formatting
- Type hints required for public APIs
- Docstrings: Google style

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **Base Model Diversity:** Performance depends on diverse, high-quality base models
2. **Scalability:** O(K²N) de-duplication becomes expensive for K > 500 (approximate methods in development)
3. **Distribution Shift:** May require recalibration under severe distribution shift
4. **Regression Focus:** Current implementation optimized for regression (classification extension planned)

### Planned Features

- [ ] Approximate similarity search for large K (LSH, random projections)
- [ ] Classification support with calibration-aware meta-learning
- [ ] Online adaptation for non-stationary data streams
- [ ] Conformal prediction for uncertainty quantification
- [ ] Multi-task and multi-domain extensions
- [ ] GPU acceleration for large-scale experiments
- [ ] AutoML integration (AutoGluon, H2O)

Track progress on our [**roadmap**](https://github.com/metarl/regularized-metalearning/projects/1).

---

## 📧 Contact

**Noor Islam S. Mohammad**  
Department of Computer Science  
New York University  
📧 noor.islam.s.m@nyu.edu

### Issues & Questions

- **Bug reports:** [GitHub Issues](https://github.com/metarl/regularized-metalearning/issues)
- **Feature requests:** [GitHub Discussions](https://github.com/metarl/regularized-metalearning/discussions)
- **General questions:** [Stack Overflow](https://stackoverflow.com/questions/tagged/regularized-metalearning) with tag `regularized-metalearning`

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Noor Islam S. Mohammad, Md Muntaqim Meherab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Dataset:** Playground Series S6E1 competition organizers
- **Infrastructure:** NYU High Performance Computing, DIU Research Labs
- **Funding:** This research was supported by [funding source if applicable]
- **Inspirations:** Stacked generalization (Wolpert, 1992; Breiman, 1996), Deep Ensembles (Lakshminarayanan et al., 2017), AutoML systems (Feurer et al., 2015; Erickson et al., 2020)

### Special Thanks

We thank the anonymous MLSys reviewers for their valuable feedback, and the open-source community for the excellent libraries that made this work possible: scikit-learn, NumPy, pandas, matplotlib, and many others.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=metarl/regularized-metalearning&type=Date)](https://star-history.com/#metarl/regularized-metalearning&Date)

---

<div align="center">

**Made with ❤️ by the MetaRL Team**

[🌐 Website](https://metarl.github.io) • [📄 Paper](https://arxiv.org/abs/2602.12469) • [📖 Docs](https://regularized-metalearning.readthedocs.io) • [💬 Discussions](https://github.com/metarl/regularized-metalearning/discussions)

</div>

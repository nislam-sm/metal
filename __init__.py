"""
Regularized Meta-Learning for Improved Generalization

A production-ready framework for ensemble learning with redundancy-aware 
projection, meta-feature augmentation, and regularized meta-learning.

Authors: Noor Islam S. Mohammad, Md Muntaqim Meherab
Paper: https://arxiv.org/abs/2602.12469
"""

__version__ = "1.0.0"
__author__ = "Noor Islam S. Mohammad, Md Muntaqim Meherab"
__email__ = "noor.islam.s.m@nyu.edu"

from .core import RegularizedMetaLearner
from .redundancy import RedundancyProjector
from .augmentation import MetaFeatureAugmenter
from .regularizers import RidgeMetaLearner, LassoMetaLearner, ElasticNetMetaLearner
from .blending import MetaEnsembleBlender
from .utils import compute_metrics, validate_predictions

__all__ = [
    "RegularizedMetaLearner",
    "RedundancyProjector",
    "MetaFeatureAugmenter",
    "RidgeMetaLearner",
    "LassoMetaLearner",
    "ElasticNetMetaLearner",
    "MetaEnsembleBlender",
    "compute_metrics",
    "validate_predictions",
]

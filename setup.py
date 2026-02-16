"""Setup script for regularized-metalearning package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="regularized-metalearning",
    version="1.0.0",
    author="Noor Islam S. Mohammad, Md Muntaqim Meherab",
    author_email="noor.islam.s.m@nyu.edu",
    description="Regularized Meta-Learning for Improved Generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/metarl/regularized-metalearning",
    project_urls={
        "Bug Tracker": "https://github.com/metarl/regularized-metalearning/issues",
        "Documentation": "https://regularized-metalearning.readthedocs.io",
        "Paper": "https://arxiv.org/abs/2602.12469",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    keywords=[
        "machine learning",
        "ensemble learning",
        "meta-learning",
        "stacking",
        "regularization",
        "deep ensembles",
        "MLSys",
    ],
)

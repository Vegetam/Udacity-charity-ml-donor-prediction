"""
Charity ML Donor Prediction Package

A machine learning package for predicting donor income levels
to optimize fundraising campaigns.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .predictor import IncomePredictor
from .model_training import train_models, optimize_model
from .data_preprocessing import preprocess_data

__all__ = [
    "IncomePredictor",
    "train_models",
    "optimize_model",
    "preprocess_data",
]

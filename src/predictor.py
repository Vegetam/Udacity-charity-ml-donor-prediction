"""
Predictor module for CharityML donor income prediction.

Provides easy-to-use interface for making predictions.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Union, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncomePredictor:
    """
    Wrapper class for income prediction model.
    
    Provides convenient methods for loading models and making predictions.
    """
    
    def __init__(self, model=None):
        """
        Initialize predictor.
        
        Args:
            model: Trained sklearn model (optional)
        """
        self.model = model
        self.feature_names = None
        
    @classmethod
    def load(cls, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            IncomePredictor instance with loaded model
        """
        logger.info(f"Loading model from {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        predictor = cls(model=model)
        logger.info("Model loaded successfully")
        return predictor
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save model file
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        logger.info(f"Saving model to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info("Model saved successfully")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict]) -> Union[int, np.ndarray]:
        """
        Predict income level.
        
        Args:
            X: Features (DataFrame, array, or dict)
            
        Returns:
            Prediction (0 for <=50K, 1 for >50K)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Handle different input types
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        prediction = self.model.predict(X)
        
        if len(prediction) == 1:
            return int(prediction[0])
        return prediction
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict]) -> Union[float, np.ndarray]:
        """
        Predict probability of high income (>50K).
        
        Args:
            X: Features (DataFrame, array, or dict)
            
        Returns:
            Probability of income >50K
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Handle different input types
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        proba = self.model.predict_proba(X)[:, 1]
        
        if len(proba) == 1:
            return float(proba[0])
        return proba
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Make prediction with confidence level.
        
        Args:
            X: Features (DataFrame or dict)
            
        Returns:
            Dictionary with prediction, probability, and confidence level
        """
        prediction = self.predict(X)
        probability = self.predict_proba(X)
        
        # Determine confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "High"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        result = {
            'prediction': '>50K' if prediction == 1 else '<=50K',
            'probability': float(probability),
            'confidence': confidence,
            'recommendation': self._get_recommendation(probability)
        }
        
        return result
    
    def _get_recommendation(self, probability: float) -> str:
        """
        Get fundraising recommendation based on prediction.
        
        Args:
            probability: Probability of income >50K
            
        Returns:
            Recommendation string
        """
        if probability >= 0.8:
            return "High priority: Request large donation"
        elif probability >= 0.6:
            return "Medium priority: Standard outreach"
        elif probability >= 0.4:
            return "Low priority: Focus on cultivation"
        else:
            return "Very low priority: Consider excluding from campaign"
    
    def batch_predict(self, X: pd.DataFrame, return_proba: bool = False) -> pd.DataFrame:
        """
        Make predictions for multiple samples.
        
        Args:
            X: Features DataFrame
            return_proba: Whether to include probabilities
            
        Returns:
            DataFrame with predictions and optional probabilities
        """
        predictions = self.predict(X)
        
        results = pd.DataFrame({
            'prediction': ['<=50K' if p == 0 else '>50K' for p in predictions]
        })
        
        if return_proba:
            probabilities = self.predict_proba(X)
            results['probability'] = probabilities
            results['confidence'] = results['probability'].apply(
                lambda p: 'High' if p >= 0.8 or p <= 0.2 else 
                         'Medium' if p >= 0.6 or p <= 0.4 else 'Low'
            )
        
        return results
    
    def get_feature_importance(self, feature_names=None) -> pd.DataFrame:
        """
        Get feature importances from the model.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            DataFrame with features and their importances
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importances")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df


def main():
    """Example usage of the predictor."""
    
    # Load trained model
    predictor = IncomePredictor.load('models/adaboost_optimized.pkl')
    
    # Example 1: Single prediction with dict
    sample = {
        'age': 35,
        'education-num': 13,
        'hours-per-week': 40,
        'capital-gain': 5000,
    }
    
    result = predictor.predict_with_confidence(sample)
    print("\nExample 1 - Single Prediction:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Example 2: Batch prediction
    # Load test data
    with open('data/processed/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    batch_results = predictor.batch_predict(X_test.head(10), return_proba=True)
    print("\nExample 2 - Batch Predictions:")
    print(batch_results)


if __name__ == "__main__":
    main()

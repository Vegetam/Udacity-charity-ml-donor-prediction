"""
Data preprocessing module for CharityML donor prediction.

Handles data cleaning, feature engineering, and transformations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load census data from CSV file.
    
    Args:
        filepath: Path to the census CSV file
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
    return data


def explore_data(data: pd.DataFrame) -> dict:
    """
    Explore basic statistics of the dataset.
    
    Args:
        data: Census DataFrame
        
    Returns:
        Dictionary with exploration statistics
    """
    n_records = len(data)
    n_greater_50k = len(data[data['income'] == '>50K'])
    n_at_most_50k = len(data[data['income'] == '<=50K'])
    greater_percent = (n_greater_50k / n_records) * 100
    
    stats = {
        'n_records': n_records,
        'n_greater_50k': n_greater_50k,
        'n_at_most_50k': n_at_most_50k,
        'greater_percent': greater_percent
    }
    
    logger.info(f"Total records: {n_records}")
    logger.info(f"Income >50K: {n_greater_50k} ({greater_percent:.2f}%)")
    logger.info(f"Income <=50K: {n_at_most_50k} ({100-greater_percent:.2f}%)")
    
    return stats


def transform_skewed_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to skewed continuous features.
    
    Args:
        features: Feature DataFrame
        
    Returns:
        Transformed features
    """
    logger.info("Applying log transformation to skewed features")
    
    skewed = ['capital-gain', 'capital-loss']
    features_transformed = features.copy()
    features_transformed[skewed] = features[skewed].apply(lambda x: np.log(x + 1))
    
    return features_transformed


def normalize_numerical_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features to [0, 1] range.
    
    Args:
        features: Feature DataFrame
        
    Returns:
        Normalized features
    """
    logger.info("Normalizing numerical features")
    
    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    features_normalized = features.copy()
    features_normalized[numerical] = scaler.fit_transform(features[numerical])
    
    return features_normalized


def encode_categorical_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    
    Args:
        features: Feature DataFrame
        
    Returns:
        One-hot encoded features
    """
    logger.info("One-hot encoding categorical features")
    
    features_encoded = pd.get_dummies(features)
    
    logger.info(f"Features after encoding: {len(features_encoded.columns)}")
    
    return features_encoded


def encode_target(income: pd.Series) -> pd.Series:
    """
    Encode income target to binary values.
    
    Args:
        income: Income series
        
    Returns:
        Encoded income (0 for <=50K, 1 for >50K)
    """
    logger.info("Encoding target variable")
    
    income_encoded = income.map({'<=50K': 0, '>50K': 1})
    
    return income_encoded


def preprocess_data(data: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 0) -> tuple:
    """
    Complete preprocessing pipeline for census data.
    
    Args:
        data: Raw census DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing pipeline")
    
    # Split features and target
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    
    # Apply transformations
    features_log = transform_skewed_features(features_raw)
    features_normalized = normalize_numerical_features(features_log)
    features_final = encode_categorical_features(features_normalized)
    income_encoded = encode_target(income_raw)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, 
        income_encoded,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    logger.info("Preprocessing complete")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir: str = 'data/processed'):
    """
    Save processed data to disk.
    
    Args:
        X_train, X_test, y_train, y_test: Processed data splits
        output_dir: Directory to save processed data
    """
    import pickle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    
    with open(f'{output_dir}/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'{output_dir}/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'{output_dir}/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(f'{output_dir}/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    logger.info("Data saved successfully")


if __name__ == "__main__":
    # Example usage
    data = load_data('data/raw/census.csv')
    stats = explore_data(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    save_processed_data(X_train, X_test, y_train, y_test)

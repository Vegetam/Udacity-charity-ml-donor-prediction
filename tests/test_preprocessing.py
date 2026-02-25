"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    explore_data,
    transform_skewed_features,
    normalize_numerical_features,
    encode_categorical_features,
    encode_target,
)


@pytest.fixture
def sample_data():
    """Create sample census data for testing."""
    data = pd.DataFrame({
        'age': [25, 30, 45, 50],
        'workclass': ['Private', 'Government', 'Private', 'Self-emp'],
        'education_level': ['Bachelors', 'Masters', 'HS-grad', 'Doctorate'],
        'education-num': [13, 14, 9, 16],
        'marital-status': ['Single', 'Married', 'Divorced', 'Married'],
        'occupation': ['Tech', 'Prof', 'Sales', 'Exec'],
        'relationship': ['Single', 'Husband', 'Single', 'Wife'],
        'race': ['White', 'Black', 'Asian', 'White'],
        'sex': ['Male', 'Female', 'Male', 'Female'],
        'capital-gain': [0, 5000, 0, 15000],
        'capital-loss': [0, 0, 500, 0],
        'hours-per-week': [40, 45, 35, 50],
        'native-country': ['US', 'US', 'India', 'US'],
        'income': ['<=50K', '>50K', '<=50K', '>50K']
    })
    return data


def test_explore_data(sample_data):
    """Test data exploration function."""
    stats = explore_data(sample_data)
    
    assert stats['n_records'] == 4
    assert stats['n_greater_50k'] == 2
    assert stats['n_at_most_50k'] == 2
    assert stats['greater_percent'] == 50.0


def test_transform_skewed_features(sample_data):
    """Test log transformation of skewed features."""
    features = sample_data.drop('income', axis=1)
    transformed = transform_skewed_features(features)
    
    # Check that transformation was applied
    assert 'capital-gain' in transformed.columns
    assert 'capital-loss' in transformed.columns
    
    # Check that log transformation produces expected results
    assert transformed['capital-gain'].iloc[0] == np.log(0 + 1)  # log(1) = 0
    assert transformed['capital-gain'].iloc[1] > 0  # log(5001) > 0


def test_normalize_numerical_features(sample_data):
    """Test normalization of numerical features."""
    features = sample_data.drop('income', axis=1)
    normalized = normalize_numerical_features(features)
    
    # Check that values are in [0, 1] range
    numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numerical_cols:
        assert normalized[col].min() >= 0
        assert normalized[col].max() <= 1


def test_encode_categorical_features(sample_data):
    """Test one-hot encoding of categorical features."""
    features = sample_data.drop('income', axis=1)
    encoded = encode_categorical_features(features)
    
    # Check that encoded features have more columns
    assert len(encoded.columns) > len(features.columns)
    
    # Check that categorical columns were expanded
    assert any('workclass_' in col for col in encoded.columns)
    assert any('education_level_' in col for col in encoded.columns)


def test_encode_target(sample_data):
    """Test income target encoding."""
    income = sample_data['income']
    encoded = encode_target(income)
    
    assert len(encoded) == 4
    assert encoded.iloc[0] == 0  # <=50K -> 0
    assert encoded.iloc[1] == 1  # >50K -> 1
    assert encoded.dtype == np.int64 or encoded.dtype == np.int32


def test_encode_target_all_values(sample_data):
    """Test that encoding produces correct values."""
    income = sample_data['income']
    encoded = encode_target(income)
    
    # Check that only 0 and 1 are present
    assert set(encoded.unique()) == {0, 1}
    
    # Check counts
    assert (encoded == 0).sum() == 2
    assert (encoded == 1).sum() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

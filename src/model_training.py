"""
Model training module for CharityML donor prediction.

Implements training, evaluation, and optimization of ML models.
"""

import numpy as np
from time import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    """
    Train a model and predict on test data.
    
    Args:
        learner: Sklearn classifier instance
        sample_size: Number of training samples to use
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        Dictionary with performance metrics
    """
    results = {}
    
    # Train the model
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start
    
    # Make predictions
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    results['pred_time'] = end - start
    
    # Calculate metrics
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
    
    logger.info(f"{learner.__class__.__name__} trained on {sample_size} samples")
    logger.info(f"  Accuracy: {results['acc_test']:.4f}, F-score: {results['f_test']:.4f}")
    
    return results


def train_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple classification models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        Dictionary with results for all models
    """
    logger.info("Initializing classifiers")
    
    # Initialize models
    clf_A = RandomForestClassifier(random_state=42)
    clf_B = AdaBoostClassifier(random_state=42)
    clf_C = LogisticRegression(random_state=42, max_iter=1000)
    
    # Calculate sample sizes
    samples_100 = len(y_train)
    samples_10 = int(0.1 * samples_100)
    samples_1 = int(0.01 * samples_100)
    
    logger.info(f"Training sizes: 1%={samples_1}, 10%={samples_10}, 100%={samples_100}")
    
    # Train all models
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(
                clf, samples, X_train, y_train, X_test, y_test
            )
    
    return results


def optimize_model(clf, X_train, y_train, X_test, y_test, parameters):
    """
    Optimize model hyperparameters using grid search.
    
    Args:
        clf: Classifier instance
        X_train, y_train: Training data
        X_test, y_test: Testing data
        parameters: Dictionary of parameters to search
        
    Returns:
        Tuple of (best_estimator, best_params, results_dict)
    """
    logger.info("Starting grid search optimization")
    logger.info(f"Parameters to search: {parameters}")
    
    # Create scorer
    scorer = make_scorer(fbeta_score, beta=0.5)
    
    # Perform grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=5, n_jobs=-1)
    grid_fit = grid_obj.fit(X_train, y_train)
    
    # Get best model
    best_clf = grid_fit.best_estimator_
    
    logger.info(f"Best parameters: {grid_fit.best_params_}")
    
    # Evaluate unoptimized model
    predictions_unopt = clf.fit(X_train, y_train).predict(X_test)
    acc_unopt = accuracy_score(y_test, predictions_unopt)
    f_unopt = fbeta_score(y_test, predictions_unopt, beta=0.5)
    
    # Evaluate optimized model
    predictions_opt = best_clf.predict(X_test)
    acc_opt = accuracy_score(y_test, predictions_opt)
    f_opt = fbeta_score(y_test, predictions_opt, beta=0.5)
    
    results = {
        'unoptimized': {'accuracy': acc_unopt, 'fscore': f_unopt},
        'optimized': {'accuracy': acc_opt, 'fscore': f_opt},
        'improvement': {
            'accuracy': acc_opt - acc_unopt,
            'fscore': f_opt - f_unopt
        }
    }
    
    logger.info("Unoptimized model:")
    logger.info(f"  Accuracy: {acc_unopt:.4f}, F-score: {f_unopt:.4f}")
    logger.info("Optimized model:")
    logger.info(f"  Accuracy: {acc_opt:.4f}, F-score: {f_opt:.4f}")
    logger.info(f"Improvement: Acc +{results['improvement']['accuracy']:.4f}, " +
                f"F-score +{results['improvement']['fscore']:.4f}")
    
    return best_clf, grid_fit.best_params_, results


def calculate_naive_predictor_score(y_test, n_greater_50k, n_records):
    """
    Calculate baseline metrics for a naive predictor.
    
    Args:
        y_test: Test labels
        n_greater_50k: Number of high-income individuals in full dataset
        n_records: Total number of records in full dataset
        
    Returns:
        Dictionary with accuracy and F-score
    """
    # Naive predictor always predicts >50K
    accuracy = n_greater_50k / n_records
    
    # For F-score calculation
    beta = 0.5
    recall = 1.0  # We predict all as >50K, so recall for >50K is 1.0
    precision = n_greater_50k / n_records
    fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    results = {'accuracy': accuracy, 'fscore': fscore}
    
    logger.info("Naive Predictor (always predicts >50K):")
    logger.info(f"  Accuracy: {accuracy:.4f}, F-score: {fscore:.4f}")
    
    return results


def save_model(model, filepath: str):
    """
    Save trained model to disk.
    
    Args:
        model: Trained sklearn model
        filepath: Path to save the model
    """
    logger.info(f"Saving model to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Model saved successfully")


def load_model(filepath: str):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded sklearn model
    """
    logger.info(f"Loading model from {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model


if __name__ == "__main__":
    import pickle
    
    # Load preprocessed data
    logger.info("Loading preprocessed data")
    with open('data/processed/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/processed/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Optimize best model (AdaBoost)
    clf = AdaBoostClassifier(random_state=42)
    parameters = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 1.0, 1.5]
    }
    
    best_model, best_params, opt_results = optimize_model(
        clf, X_train, y_train, X_test, y_test, parameters
    )
    
    # Save best model
    save_model(best_model, 'models/adaboost_optimized.pkl')

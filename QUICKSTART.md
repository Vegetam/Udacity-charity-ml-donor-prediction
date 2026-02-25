# Quickstart Guide

## Setup

1. **Install requirements** (requires Python 3.9–3.13):
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter**:
   ```bash
   cd notebooks
   jupyter notebook finding_donors.ipynb
   ```
   > The notebook must be launched from inside the `notebooks/` folder so it can find `visuals.py` and the relative path to `../data/raw/census.csv`.

## Running the Notebook

Run cells **top to bottom** with **Shift+Enter**. The key sections are:

| Section | What it does |
|---|---|
| **Exploring the Data** | Loads census data, computes n_records, n_greater_50k, etc. |
| **Preparing the Data** | Log-transforms skewed features, normalizes, one-hot encodes |
| **Naive Predictor** | Computes baseline accuracy & F-score (Q1) |
| **Model Pipeline** | Trains & evaluates Random Forest, AdaBoost, Logistic Regression |
| **Grid Search Tuning** | Optimizes AdaBoost with GridSearchCV (takes ~2–5 min) |
| **Feature Importance** | Extracts and visualizes top 5 predictive features |
| **Reduced Model** | Re-trains on top 5 features only and compares performance |

## After Running

Fill in the **Question 5 results table** with the accuracy and F-score values printed by the grid search cell.

## Notes

- The grid search cell (9 combinations × 5-fold CV = 45 fits) may take **2–5 minutes** on a modern laptop.
- Python version used: **3.13+**

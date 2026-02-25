# 🎯 Predicting Donor Income for CharityML

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project that predicts whether individuals earn more than $50,000 annually using the 1994 U.S. Census dataset. Built to help CharityML identify potential donors and optimize fundraising efforts.

![Project Banner](https://img.shields.io/badge/ML-Supervised%20Learning-brightgreen) ![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

CharityML is a fictitious charity organization that needs to identify individuals most likely to donate. This project uses supervised machine learning to predict income levels based on census data, enabling targeted fundraising campaigns.

**Problem Statement**: Given demographic and employment data, predict whether an individual's income exceeds $50,000/year.

**Solution**: Implemented and compared multiple classification algorithms, optimized the best performer (AdaBoost), and achieved **87% accuracy** with an **F-score of 0.76**.

---

## ✨ Key Features

- 🔍 **Comprehensive Data Analysis**: Exploratory data analysis with visualization
- 🤖 **Multiple ML Models**: Random Forest, AdaBoost, and Logistic Regression
- ⚡ **Hyperparameter Tuning**: Grid search optimization with cross-validation
- 📊 **Feature Importance**: Identified top 5 predictive features
- 🎯 **Model Selection**: Systematic evaluation based on performance and computational cost
- 📉 **Feature Reduction**: Achieved 95% performance with only 5 features (10-20x faster)
- 📈 **Performance Visualization**: Clear charts comparing model metrics

---

## 📊 Dataset

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income)

**Original Publication**: Ron Kohavi and Barry Becker, "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid" (1996)

### Dataset Characteristics

- **Samples**: 45,222 individuals (training: 36,177 | testing: 9,045)
- **Features**: 13 attributes (demographic, employment, financial)
- **Target**: Binary classification (<=50K or >50K)
- **Class Distribution**: 76% earn ≤$50K, 24% earn >$50K

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Continuous | Age in years |
| workclass | Categorical | Employment type (Private, Government, Self-employed) |
| education_level | Categorical | Highest education completed |
| education-num | Continuous | Years of education (numerical) |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status |
| race | Categorical | Race |
| sex | Categorical | Gender |
| capital-gain | Continuous | Capital gains |
| capital-loss | Continuous | Capital losses |
| hours-per-week | Continuous | Hours worked per week |
| native-country | Categorical | Country of origin |

---

## 🤖 Models Evaluated

### 1. Random Forest Classifier
- **Purpose**: Baseline ensemble model
- **Strengths**: Handles non-linear relationships, provides feature importance
- **Performance**: Accuracy: 85.2%, F-score: 0.73

### 2. AdaBoost Classifier ⭐ **[Selected Model]**
- **Purpose**: Boosting ensemble for improved accuracy
- **Strengths**: High performance, handles class imbalance well
- **Performance**: Accuracy: 86.8%, F-score: 0.76
- **Optimized**: n_estimators=100, learning_rate=1.0

### 3. Logistic Regression
- **Purpose**: Fast linear baseline
- **Strengths**: Interpretable, quick training
- **Performance**: Accuracy: 84.1%, F-score: 0.71

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | F-Score (β=0.5) | Training Time | Prediction Time |
|-------|----------|-----------------|---------------|-----------------|
| Naive Predictor | 24.8% | 0.292 | - | - |
| Logistic Regression | 84.1% | 0.710 | 0.15s | 0.01s |
| Random Forest | 85.2% | 0.730 | 2.10s | 0.08s |
| **AdaBoost (Unoptimized)** | 86.3% | 0.745 | 1.85s | 0.05s |
| **AdaBoost (Optimized)** | **86.8%** | **0.760** | 2.20s | 0.05s |

### Feature Importance (Top 5)

1. **Capital-gain** (0.245) - Investment income
2. **Age** (0.189) - Career progression indicator
3. **Hours-per-week** (0.156) - Work commitment level
4. **Education-num** (0.132) - Education level
5. **Marital-status_Married-civ-spouse** (0.098) - Economic stability

### Reduced Feature Model

Using only the top 5 features:
- **Accuracy**: 84.9% (only 1.9% decrease)
- **F-Score**: 0.728 (only 3.2% decrease)
- **Training Time**: **10-20x faster**
- **Recommended for production** due to speed/performance trade-off

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/charity-ml-donor-prediction.git
cd charity-ml-donor-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## 💻 Usage

### Run Jupyter Notebook

```bash
jupyter notebook notebooks/finding_donors.ipynb
```

### Run Python Script

```bash
python src/train_model.py
```

### Make Predictions

```python
from src.predictor import IncomePredictor

# Load trained model
predictor = IncomePredictor.load('models/adaboost_optimized.pkl')

# Make prediction
sample = {
    'age': 35,
    'education-num': 13,
    'hours-per-week': 40,
    'capital-gain': 5000,
    'marital-status': 'Married-civ-spouse'
}

prediction = predictor.predict(sample)
probability = predictor.predict_proba(sample)

print(f"Income prediction: {'>50K' if prediction else '<=50K'}")
print(f"Confidence: {probability:.2%}")
```

---

## 📁 Project Structure

```
charity-ml-donor-prediction/
│
├── data/
│   ├── raw/
│   │   └── census.csv              # Original dataset
│   └── processed/
│       ├── X_train.pkl             # Processed training features
│       ├── X_test.pkl              # Processed testing features
│       ├── y_train.pkl             # Training labels
│       └── y_test.pkl              # Testing labels
│
├── notebooks/
│   ├── finding_donors.ipynb        # Main analysis notebook
│   ├── exploratory_analysis.ipynb  # EDA notebook
│   └── model_comparison.ipynb      # Model evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning and encoding
│   ├── feature_engineering.py      # Feature transformations
│   ├── model_training.py           # Model training functions
│   ├── model_evaluation.py         # Evaluation metrics
│   ├── predictor.py                # Prediction interface
│   └── utils.py                    # Helper functions
│
├── models/
│   ├── adaboost_optimized.pkl      # Best trained model
│   ├── random_forest.pkl           # Random Forest model
│   └── logistic_regression.pkl     # Logistic Regression model
│
├── reports/
│   ├── figures/                    # Generated visualizations
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   └── confusion_matrix.png
│   ├── finding_donors_report.html  # HTML report
│   └── project_summary.pdf         # Executive summary
│
├── tests/
│   ├── test_preprocessing.py       # Unit tests
│   ├── test_models.py
│   └── test_predictor.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── config.yaml                     # Configuration file
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Log transformation** of skewed features (capital-gain, capital-loss)
- **MinMax normalization** of numerical features
- **One-hot encoding** of categorical variables
- **Train-test split** (80-20 ratio)

### 2. Baseline Model
- Naive predictor (always predicts >50K)
- Accuracy: 24.8%, F-score: 0.292

### 3. Model Selection
Evaluated three algorithms on multiple training sizes (1%, 10%, 100%):
- Random Forest Classifier
- AdaBoost Classifier
- Logistic Regression

### 4. Hyperparameter Tuning
Grid search with 5-fold cross-validation:
```python
parameters = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0, 1.5]
}
```

### 5. Feature Importance Analysis
- Extracted feature importances from AdaBoost
- Identified top 5 predictive features
- Evaluated reduced feature model

### 6. Model Evaluation Metrics
- **Accuracy**: Overall correctness
- **F-score (β=0.5)**: Weighted harmonic mean (emphasizes precision)
- **Training/Prediction time**: Computational efficiency

---

## 🎯 Key Insights

### What Makes Someone Likely to Earn >$50K?

1. **Capital Gains are Crucial** (24.5% importance)
   - Investment income is the strongest predictor
   - Suggests wealth accumulation patterns

2. **Age Matters** (18.9% importance)
   - Older individuals have higher income probability
   - Reflects career progression and experience

3. **Work Hours Count** (15.6% importance)
   - Full-time (40+ hours) strongly correlates with higher income
   - Part-time workers rarely exceed $50K threshold

4. **Education Pays Off** (13.2% importance)
   - Each additional year of education increases income probability
   - Advanced degrees (Master's, Doctorate) show strong correlation

5. **Marriage is an Indicator** (9.8% importance)
   - Married individuals more likely to earn >$50K
   - May reflect dual-income households or stability factors

### Business Recommendations for CharityML

1. **Prioritize individuals with**:
   - High capital gains (investment income)
   - Advanced education (Bachelor's or higher)
   - Full-time employment (40+ hours/week)
   - Age 35+ years

2. **Use the reduced 5-feature model** for:
   - Real-time donor scoring
   - Large-scale batch processing
   - Mobile/edge deployments

3. **Allocate resources efficiently**:
   - High-confidence predictions (>80%) → Large donation requests
   - Medium-confidence (50-80%) → Standard outreach
   - Low-confidence (<50%) → Focus on cultivation

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Test XGBoost and LightGBM for better performance
- [ ] Implement neural networks for comparison
- [ ] Try stacking/ensemble methods
- [ ] Add SHAP values for better interpretability

### Feature Engineering
- [ ] Create interaction features (age × education, etc.)
- [ ] Derive new features (income bracket estimator)
- [ ] Time-series analysis if temporal data available
- [ ] Geographic clustering for native-country

### Deployment
- [ ] Build REST API with FastAPI
- [ ] Create web interface with Streamlit
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Set up CI/CD pipeline
- [ ] Add model monitoring and retraining

### Data Improvements
- [ ] Collect more recent census data (post-1994)
- [ ] Handle missing values more sophisticatedly
- [ ] Address potential bias in protected attributes
- [ ] Gather additional economic indicators

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/charity-ml-donor-prediction.git
cd charity-ml-donor-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Dataset
- **Original Authors**: Ron Kohavi and Barry Becker
- **Source**: UCI Machine Learning Repository
- **Paper**: ["Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf) (KDD-96)

### Inspiration
- Udacity Data Scientist Nanodegree Program
- scikit-learn documentation and examples
- Kaggle community tutorials

### References

1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
2. Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting." *Journal of Computer and System Sciences*, 55(1), 119-139.
3. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

---

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

**Project Link**: [https://github.com/yourusername/charity-ml-donor-prediction](https://github.com/yourusername/charity-ml-donor-prediction)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/charity-ml-donor-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/charity-ml-donor-prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/charity-ml-donor-prediction)

**Made with ❤️ and Python**

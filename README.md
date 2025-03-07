# Telecom Customer Churn Prediction

## Overview

This project develops a machine learning solution to predict customer churn for a telecommunications company. It includes data preprocessing, exploratory data analysis, feature engineering, and the training and evaluation of multiple classification models.

## Project Structure

```
churn_prediction/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── data.csv
├── logs/
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_factory.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_visualizer.py
│   │   └── model_visualizer.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
└── main.py
```

## Installation

1. Clone the repository:

```
git clone https://github.com/Midhilesh4890/Customer-Churn
cd customer-churn
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Usage

1. Telco Customer Churn dataset in `data/raw/data.csv`

2. Run the main pipeline:

```
python main.py
```

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- Ensemble (voting classifier)
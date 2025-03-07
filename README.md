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
git clone <repository-url>
cd churn_prediction
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
pip install -e .
```

## Usage

1. Place the Telco Customer Churn dataset in `data/raw/data.csv`

2. Run the main pipeline:

```
python main.py
```

3. For exploratory data analysis, run the Jupyter notebook:

```
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## Models

The project implements the following models:

- Logistic Regression
- Decision Tree
- Random Forest
- Ensemble (voting classifier)

## Results

The best-performing model achieved 79.0% accuracy with an F1 score of 61.0%. Key predictors of churn include:

- Contract type (month-to-month customers more likely to churn)
- Tenure (newer customers more likely to churn)
- Internet service type (fiber optic customers more likely to churn)
- Whether customers have online security and tech support services

## License

[MIT](LICENSE)

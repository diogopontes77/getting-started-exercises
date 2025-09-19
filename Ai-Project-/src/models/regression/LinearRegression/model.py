import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer
from xgboost import XGBClassifier

# Profit scoring
def profit_function(y_test, y_pred):
    margin = (abs(y_test - y_pred) / y_test) < 0.3
    return margin.sum() * 100 # ara cada y_pred que esteja dentor da margem de 30% adicionar 100 ao lucro total

profit_score = make_scorer(profit_function, greater_is_better=True) # quanto mais melhor e vamos usar isto como o nosso scorer na grid search

def build_pipeline(preprocessor, params=None):
    if params is None:
        rgr = LinearRegression()
    else:
        rgr = LinearRegression(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rgr)
    ])
    return pipeline

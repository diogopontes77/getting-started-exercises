import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer
from xgboost import XGBClassifier

# Profit scoring
profit_matrix = np.array([
    [0,   -200],
    [-200, 100]
])

def profit_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return (cm * profit_matrix).sum()

profit_scorer = make_scorer(profit_score, greater_is_better=True)

def build_pipeline(preprocessor, params=None):
    if params is None:
        clf = RandomForestClassifier()
    else:
        clf = RandomForestClassifier(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

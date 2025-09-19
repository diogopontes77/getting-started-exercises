import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer
from xgboost import XGBClassifier

# Profit scoring
profit_matrix = np.array([
    [0,   -200],
    [-200, 100]
])

def profit_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return (cm * profit_matrix).sum()

profit_scorer = make_scorer(profit_score, greater_is_better=True)

def build_pipeline(preprocessor, params=None):
    if params is None:
        clf = GradientBoostingClassifier(random_state=42)
    else:
        clf = GradientBoostingClassifier(**params, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipeline

import optuna
from sklearn.metrics import roc_auc_score
from model import build_pipeline  # Assumes this accepts a model name or type
from sklearn.tree import DecisionTreeClassifier

def optimize_decision_tree(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "class_weight": 'balanced',
            "random_state": 42
        }

        pipeline = build_pipeline(preprocessor, model_type="decision_tree", model_params=params)
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

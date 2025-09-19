import optuna
from sklearn.metrics import roc_auc_score
from model import build_pipeline
from sklearn.svm import SVC

def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True,
            "random_state": 42
        }

        pipeline = build_pipeline(preprocessor,params)
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

import optuna
from sklearn.metrics import roc_auc_score
from model import build_pipeline

def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }

        pipeline = build_pipeline(preprocessor, params)
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

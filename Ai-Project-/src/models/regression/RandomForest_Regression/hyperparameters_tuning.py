import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from model import build_pipeline 

def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        params = {
             "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42       
            }

        # Assumes build_pipeline(preprocessor, model_type, params)
        pipeline = build_pipeline(preprocessor,params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

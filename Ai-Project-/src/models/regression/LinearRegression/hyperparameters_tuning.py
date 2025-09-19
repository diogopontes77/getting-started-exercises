import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, roc_auc_score
from model import build_pipeline

def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        params = {
            # Linear regression nao tem hyperparameters para otimizar
        }

        pipeline = build_pipeline(preprocessor, params)
        pipeline.fit(X_train, y_train)
        y_pred= pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from model import build_pipeline
from sklearn.neural_network import MLPRegressor

def optimize_mlp_regressor(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_layer_{i}", 16, 128, step=16) 
            for i in range(trial.suggest_int("n_layers", 1, 3))
        )

        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            "max_iter": 300,
            "random_state": 42
        }

        pipeline = build_pipeline(preprocessor, model_type="mlp_regressor", model_params=params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

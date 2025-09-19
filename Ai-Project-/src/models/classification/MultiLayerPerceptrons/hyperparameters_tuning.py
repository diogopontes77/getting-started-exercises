import optuna
from sklearn.metrics import roc_auc_score
from model import build_pipeline  # assuming it builds MLP pipeline now

def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = tuple(
        trial.suggest_int(f"n_units_layer_{i}", 16, 128, step=16)
        for i in range(n_layers)
    )
        # Suggest hyperparameters for MLPClassifier
        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            "max_iter": 300        
            }

        pipeline = build_pipeline(preprocessor, params)
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

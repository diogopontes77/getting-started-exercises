import optuna
from sklearn.metrics import roc_auc_score
from model import build_pipeline
import xgboost as xgb


def optimize(X_train, y_train, X_test, y_test, preprocessor, n_trials=20):
    # Preprocess before optimization (as DMatrix needs raw arrays)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    dtrain = xgb.DMatrix(X_train_proc, label=y_train)
    dvalid = xgb.DMatrix(X_test_proc, label=y_test)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "seed": 42,
        }

        n_estimators = trial.suggest_int("n_estimators", 100, 300)

        # Train with early stopping on validation data to avoid overfitting
        evals = [(dtrain, "train"), (dvalid, "valid")]
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators,
                            evals=evals, early_stopping_rounds=20, verbose_eval=False)

        # Predict on validation set
        y_pred_proba = booster.predict(dvalid)

        # Calculate AUC score
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study

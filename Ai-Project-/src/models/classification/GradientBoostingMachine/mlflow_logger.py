import mlflow
from mlflow.models import infer_signature

def log_experiment(model, X_test_df, y_pred, best_params, metrics):
    mlflow.set_tracking_uri(uri="http://localhost:5001")
    mlflow.set_experiment("GBM Credit Model")

    with mlflow.start_run():
        mlflow.log_params(best_params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        signature = infer_signature(X_test_df, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="gbm-optuna-credit-model"
        )

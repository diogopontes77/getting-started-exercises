from train import train_model
from shap_analysis import explain_model
from mlflow_logger import log_experiment

def main():
    # 🔹 Train model & get results
    final_pipeline, (X_test, y_pred, y_pred_proba), metrics, best_params = train_model()

    print("Treino")
    print("Best Parameters:", best_params)
    print("Metrics:", metrics)

    # 🔹 Run SHAP explainability
    print("SHAP")
    X_test_df = explain_model(final_pipeline, X_test)

    # 🔹 Log to MLflow
    print("Ver o MLFLOW")
    log_experiment(final_pipeline, X_test_df, y_pred, best_params, metrics)


if __name__ == "__main__":
    main()

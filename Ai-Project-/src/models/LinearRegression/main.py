from model1.train import train_model
from model1.shap_analysis import explain_model
from model1.mlflow_logger import log_experiment

def main():
    # 🔹 Train model & get results
    final_pipeline, (X_test, y_pred, y_pred_proba), metrics, best_params = train_model()

    print("Training complete ✅")
    print("Best Parameters:", best_params)
    print("Metrics:", metrics)

    # 🔹 Run SHAP explainability
    print("Running SHAP analysis...")
    X_test_df = explain_model(final_pipeline, X_test)

    # 🔹 Log to MLflow
    print("Logging to MLflow...")
    log_experiment(final_pipeline, X_test_df, y_pred, best_params, metrics)

    print("All done 🚀")

if __name__ == "__main__":
    main()

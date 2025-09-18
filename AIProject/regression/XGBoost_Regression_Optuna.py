import re
import numpy as np
import optuna
import pandas as pd
import shap
import mlflow
from mlflow.models import infer_signature
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


df = pd.read_csv('DatasetCredit-g.csv')


y = df['credit_amount']
X = df.drop(columns=['credit_amount', 'class', 'foreign_worker']) 


df['foreign'] = df['foreign_worker'].map({'no': 0, 'yes': 1})
categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========== Profit Function ==========

def profit_function(y_true, y_pred):
    within_margin = np.abs(y_true - y_pred) / y_true < 0.3
    return 100 * within_margin.sum()   # total profit, ver isto melhor

# ========== Optuna Objective ==========
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse * -1

# ========== Run Optuna ==========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best MSE (neg):", study.best_value)
print("Best Parameters:", study.best_params)

# ========== Final Model Training ==========
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", final_model)
])
pipeline.fit(X_train, y_train)

# ========== Evaluation ==========
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Final MAE:", mae)
print("Final R²:", r2)


# ========== SHAP ==========
processed = preprocessor.fit_transform(X_test)
feature_names = preprocessor.get_feature_names_out()
feature_names = [re.sub(r"[<>\[\]]", "", name) for name in feature_names]

X_test_df = pd.DataFrame(
    processed.toarray() if hasattr(processed, 'toarray') else processed,
    columns=feature_names
)

explainer = shap.Explainer(final_model)
shap_values = explainer(X_test_df)
shap.summary_plot(shap_values, X_test_df, show=False)
plt.savefig("shap_summary_plot_regression.png")

# ========== MLflow ==========
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("XGBoost Regression Credit Model")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    signature = infer_signature(X_test_df, y_pred)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_test_df.head(),
        registered_model_name="xgb-regression-credit-model"
    )

    mlflow.log_artifact("shap_summary_plot_regression.png")

import re
from matplotlib import pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import mlflow
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('DatasetCredit-g.csv')
mlflow.set_tracking_uri(uri="http://localhost:5001")
mapping_foreign = {'no': 0, 'yes': 1}
mapping = {'bad': 0, 'good': 1}

df['class_binary'] = df['class'].map(mapping)
df['foreign'] = df['foreign_worker'].map(mapping_foreign)
df = df.drop(columns=['foreign_worker', 'class'])

X = df.drop(columns=["class_binary"])
y = df["class_binary"]

categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== Profit score ==========
profit_matrix = np.array([
    [0,   -200],
    [-200, 100]
])

def profit_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return (cm * profit_matrix).sum()

profit_scorer = make_scorer(profit_score, greater_is_better=True)

# ========== Optuna ==========
def objective(trial):
    clf = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 10),
        eval_metric='logloss',
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

# ========== Run Optuna ==========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best AUC:", study.best_value)
print("Best Parameters:", study.best_params)

# ========== Final model training ==========
best_params = study.best_params
clf = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

final_pipeline.fit(X_train, y_train)

# ========== Evaluation ==========
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
profit = profit_score(y_test, y_pred)

print("Final Accuracy:", accuracy)
print("Final AUC:", auc)
print("Final Profit:", profit)

# ========== SHAP n ==========
fitted_model = final_pipeline.named_steps["classifier"]
fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
X_test_processed = fitted_preprocessor.transform(X_test)
raw_feature_names = fitted_preprocessor.get_feature_names_out()
feature_names = [re.sub(r"[<>\[\]]", "", name) for name in raw_feature_names]

X_test_df = pd.DataFrame(
    X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
    columns=feature_names
)

explainer = shap.TreeExplainer(fitted_model)
shap_values = explainer.shap_values(X_test_df)
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)
plt.savefig("shap_summary_plot_XGBoost.png")

# ========== MLflow ==========
mlflow.set_experiment("XGBoost Optuna Credit Model")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("profit", profit)

    signature = infer_signature(X_test_df, y_pred)

    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_test_df.head(),
        registered_model_name="xgb-optuna-credit-model"
    )

    mlflow.log_artifact("shap_summary_plot.png")

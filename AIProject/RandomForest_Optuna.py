from matplotlib import pyplot as plt
import optuna
import pandas as pd
import shap
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('DatasetCredit-g.csv')
#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
shap.initjs()
mapping_foreign = {'no': 0, 'yes': 1}
mapping = {'bad': 0, 'good': 1}

#mlflow.set_experiment("Getting Started Exercise CCG")


df['class_binary'] = df['class'].map(mapping)
df['foreign'] = df['foreign_worker'].map(mapping_foreign)
df = df.drop(columns=['foreign_worker', 'class'])

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

# Testar com xgboost depois disto e meter isto mais organizado
profit_matrix = np.array([
    [0,   -200],
    [-200, 100]
])

def profit_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) # cria a confusion matrix, dps multiplica pela profit matrix que é definida no read me do github
    return (cm * profit_matrix).sum() # soma final sendo o nosso objetivo ter o maximo possivel de lucro

profit_scorer = make_scorer(profit_score, greater_is_better=True) # quanto mais melhor e vamos usar isto como o nosso scorer na grid search


data_x = df.drop(columns=["class_binary"])
X= data_x
y= df["class_binary"]

categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

# One htot encoder para as features categoricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Weight balance dá mais importancia a classes minoritarias para mitigar os efeitos de class imbalance, dando
# melhor performance nas classes menos representadas, aumentado recall e precision nestas classes e neste caso, aumentando o lucro
# para isso metemos class_Weight='balanced'

#rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#X_train_processed = preprocessor.fit_transform(X_train)
#X_test_processed = preprocessor.transform(X_test)
#rf_classifier.fit(X_train_processed, y_train)

#feature_names = preprocessor.get_feature_names_out()

#X_test_processed_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
                                   #columns=feature_names)
# Temos que utilizar Dataframes, nao numpy array, basicamente temos que mudar a estrutura do numpy array para um dataframe
#y_pred = rf_classifier.predict(X_test_processed)

#print(classification_report(y_pred, y_test))

# Nos nao priorizamos a acuracia, mas sim o lucro
from sklearn.model_selection import GridSearchCV

def objective(trial):
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 50),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        class_weight='balanced',
        random_state=42
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    feature_names = preprocessor.get_feature_names_out()
    
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

# Use the profit_scorer defined earlier
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best AUC:", study.best_value)
print("Best Parameters:", study.best_params)
#grid_search.fit(X_train_processed, y_train)
explainer = shap.TreeExplainer(rf_classifier)

# # Get SHAP values for the test set
shap_values = explainer.shap_values(X_test_processed)
print(shap_values)
# # SHAP values for class 1 (good clients in your case)
shap_values_class_1 = shap_values[1]

# # Plot the SHAP summary plot for class 1
shap.summary_plot(shap_values_class_1, X_test_processed)


shap.summary_plot(shap_values_class_1, X_test_processed, feature_names=preprocessor.get_feature_names_out())

plt.savefig("shap_summary_plot.png")
accuracy = accuracy_score(y_test, grid_search.predict(X_test_processed))

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)
print("Best Profit Score:", grid_search.best_score_)
print("Accuracy:", accuracy)
#Má previsao e recall para valores 0, ver o que posso fazer para melhorar

# https://www.mlflow.org/docs/latest/ml/tracking/quickstart/
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(param_grid_search)

#     # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("Profit", grid_search.best_score_)

#     # Infer the model signature
    signature = infer_signature(
    pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
                 columns=feature_names),
    rf_classifier.predict(X_train_processed)
)

#     # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=rf_classifier,
        signature=signature,
        artifact_path="model",
        input_example=pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
                           columns=feature_names),
        registered_model_name="random-forest-getting-started",
    )

#     # Set a tag that we can use to remind ourselves what this model was for
#     # mlflow.set_logged_model_tags(
#     #     model_info.model_id, {"Training Info": "Basic Model random forest classifier with grid search"}
#     # )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test_processed_df)

# Create the SHAP explainer object


feature_names = list(df.columns.values)

#result = pd.DataFrame(X_test, columns=feature_names)
#result["actual_class"] = y_test
#result["predicted_class"] = predictions

#result[:4]
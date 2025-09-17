from matplotlib import pyplot as plt
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

df = pd.read_csv('DatasetCredit-g.csv')
#mlflow.set_tracking_uri(uri="http://localhost:8080")

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

xgb_classifier = XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
xgb_classifier.fit(X_train_processed, y_train)

feature_names = preprocessor.get_feature_names_out()

X_test_processed_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
                                   columns=feature_names)
# Temos que utilizar Dataframes, nao numpy array, basicamente temos que mudar a estrutura do numpy array para um dataframe
y_pred = xgb_classifier.predict(X_test_processed)

print(classification_report(y_pred, y_test))

# Nos nao priorizamos a acuracia, mas sim o lucro
from sklearn.model_selection import GridSearchCV

params_random_search = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ]
}

# Use the profit_scorer defined earlier
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params_random_search, n_iter=5, scoring=profit_scorer, n_jobs=-1, cv=3, verbose=3, random_state=42)
random_search.fit(X_train_processed, y_train)
#grid_search.fit(X_train_processed, y_train)

explainer = shap.TreeExplainer(xgb_classifier)

# # Get SHAP values for the test set
shap_values = explainer.shap_values(X_test_processed)
#print(shap_values)
# # SHAP values for class 1 (good clients in your case)

# # Plot the SHAP summary plot for class 1
shap.summary_plot(shap_values, X_test_processed)

# NO xgboost não é preciso definir classes
shap.summary_plot(shap_values, X_test_processed, feature_names=preprocessor.get_feature_names_out())

plt.savefig("shap_summary_plot_XGBoost.png")

accuracy = accuracy_score(y_test, random_search.predict(X_test_processed))

print("Best Parameters:", random_search.best_params_)
print("Best Estimator:", random_search.best_estimator_)
print("Best Profit Score:", random_search.best_score_)
print("Accuracy:", accuracy)
#Má previsao e recall para valores 0, ver o que posso fazer para melhorar

# https://www.mlflow.org/docs/latest/ml/tracking/quickstart/
# with mlflow.start_run():
#     # Log the hyperparameters
#     mlflow.log_params(param_grid_search)

#     # Log the loss metric
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("Profit", grid_search.best_score_)

#     # Infer the model signature
#     signature = infer_signature(
#     pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
#                  columns=feature_names),
#     xgb_classifier.predict(X_train_processed)
# )

#     # Log the model, which inherits the parameters and metric
#     model_info = mlflow.sklearn.log_model(
#         sk_model=xgb_classifier,
#         signature=signature,
#         artifact_path="model",
#         input_example=pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
#                            columns=feature_names),
#         registered_model_name="random-forest-getting-started",
#     )

#     # Set a tag that we can use to remind ourselves what this model was for
#     # mlflow.set_logged_model_tags(
#     #     model_info.model_id, {"Training Info": "Basic Model random forest classifier with grid search"}
#     # )

#     loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# predictions = loaded_model.predict(X_test_processed_df)

# feature_names = list(df.columns.values)

# result = pd.DataFrame(X_test, columns=feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions

# result[:4]
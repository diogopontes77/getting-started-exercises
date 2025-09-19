from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from dataloader import load_data
from preprocessing import build_preprocessor
from model import build_pipeline, profit_score
from hyperparameters_tuning import optimize
import xgboost as xgb

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



    preprocessor = build_preprocessor(X)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    dtrain = xgb.DMatrix(X_train_proc, label=y_train)
    dtest = xgb.DMatrix(X_test_proc, label=y_test)
    
    study = optimize(X_train, y_train, X_test, y_test, preprocessor)

    best_params = study.best_params
    
    num_boost_round = best_params.pop("n_estimators", 100)
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42
    })
    
    final_model = xgb.train(best_params, dtrain, num_boost_round=num_boost_round)
    
    y_pred_proba = final_model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    print("AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # final_pipeline = build_pipeline(preprocessor, best_params)
    # final_pipeline.fit(X_train, y_train)

    # y_pred = final_pipeline.predict(X_test)
    # y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
         "accuracy": accuracy_score(y_test, y_pred),
         "auc": roc_auc_score(y_test, y_pred_proba),
         "profit": profit_score(y_test, y_pred)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))

    
    # cm = confusion_matrix(y_test, y_pred)
    
    # return final_pipeline, (X_test, y_pred, y_pred_proba), metrics, best_params, cm
    
    return final_model, (X_test, y_pred, y_pred_proba), best_params,metrics, cm

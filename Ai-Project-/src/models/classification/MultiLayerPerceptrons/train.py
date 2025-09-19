from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from dataloader import load_data
from preprocessing import build_preprocessor
from model import build_pipeline, profit_score
from hyperparameters_tuning import optimize

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    preprocessor = build_preprocessor(X)
    study = optimize(X_train, y_train, X_test, y_test, preprocessor)

    best_params = study.best_params
    final_pipeline = build_pipeline(preprocessor, best_params)
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "profit": profit_score(y_test, y_pred)
    }
    
    print(classification_report(y_test, y_pred))

    
    return final_pipeline, (X_test, y_pred, y_pred_proba), metrics, best_params, cm

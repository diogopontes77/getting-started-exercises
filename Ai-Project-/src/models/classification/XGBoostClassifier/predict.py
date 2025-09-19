def predict(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]

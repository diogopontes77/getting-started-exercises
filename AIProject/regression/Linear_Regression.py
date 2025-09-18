import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb



df = pd.read_csv('DatasetCredit-g.csv')

data_x = df.drop(columns=["credit_amount"])
X= data_x
y= df["credit_amount"]


def profit_function(y_test, y_pred):
    margin = (abs(y_test - y_pred) / y_test) < 0.3
    return margin.sum() * 100 # ara cada y_pred que esteja dentor da margem de 30% adicionar 100 ao lucro total

profit_score = make_scorer(profit_function, greater_is_better=True) # quanto mais melhor e vamos usar isto como o nosso scorer na grid search

categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


params = {"objective": "reg:squarederror", "tree_method": "hist"}

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


#model = Linear Regression
model = LinearRegression()

#model.fit(X_train_processed, y_train)
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
#print(f"Accuracy: {accuracy:.2f}")
print(f"R²: {r2:.2f}")
#print(classification_report(y_pred, y_test))
profit = profit_function(y_test, y_pred)
print("Profit:", profit)
import h2o
from h2o.automl import H2OAutoML
h2o.init()
#REGRESSOR, FAZER O MESMO PARA CLASSIFICADOR
df = h2o.import_file("DatasetCredit-g.csv")

df['foreign'] = df['foreign_worker'].asfactor().map({'no': 0, 'yes': 1})
df['class'] = df['class'].asfactor().map({'bad': 0,
                                            'good': 1})
X = df.columns
y = "credit_amount"
X.remove(y)
X.remove('foreign_worker')
X.remove('class')

train, test = df.split_frame(ratios=[0.7], seed=42)

# Start AutoML
aml = H2OAutoML(max_models=10, max_runtime_secs=300, seed=42, sort_metric="RMSE")
aml.train(x=X, y=y, training_frame=train)

# View leaderboard
print(aml.leaderboard)

# Make predictions on test set
preds = aml.leader.predict(test)
print(preds.head())


leader_model = aml.leader

# # Get SHAP values (contributions) for test set
# shap_values = leader_model.predict_contributions(test)

# # View the first few rows of SHAP values
# print(shap_values.head())

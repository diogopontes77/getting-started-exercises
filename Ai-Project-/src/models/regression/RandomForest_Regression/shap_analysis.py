import re
import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_model(pipeline, X_test):
    fitted_model = pipeline.named_steps["regressor"]
    fitted_preprocessor = pipeline.named_steps["preprocessor"]

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
    plt.savefig("shap_summary_plot_RF_Regression.png")

    return X_test_df

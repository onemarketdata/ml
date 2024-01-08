import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
import shap


def add_lags(df, columns, periods=[1, 2, 3, 4, 38, 39, 78, 195]):
    lags_columns = []
    for column in columns:
        for lag in periods:
            feature_col_name = f"{column}_lag_{lag}"
            df[feature_col_name] = df.shift(lag)[column]
            lags_columns.append(feature_col_name)

    return df, lags_columns

def evaluate_feature_importance(df, train_indexes, lags_columns, target, model=None, threshold=0.95, plot=False):
    if model is None:
        warnings.warn("Feature Importance: model is not specified, using CatBoostRegressor", UserWarning)
        model = CatBoostRegressor()
    model.fit(df.loc[train_indexes, lags_columns],
            df.loc[train_indexes, target],
            verbose=0,
            plot=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.loc[train_indexes, lags_columns])

    feature_importance = pd.DataFrame(shap_values, columns=lags_columns).abs().sum().sort_values(ascending=False)
    feature_importance_normalized = feature_importance / feature_importance.sum()
    cumulative_importance = feature_importance_normalized.cumsum()
    idx = np.where(cumulative_importance > threshold)[0][0]

    if plot:
        shap.summary_plot(shap_values, df.loc[train_indexes, lags_columns])
        plt.plot(range(len(cumulative_importance)), cumulative_importance)
        plt.axvline(x=idx, color='g', linestyle='--', label=f'Optimal Number of Features ({idx})')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.legend()
        plt.show()

    top_features = feature_importance[cumulative_importance <= threshold].index
    return top_features


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import DMatrix

from catboost import Pool, cv as cat_cv
from xgboost import DMatrix, cv as xgb_cv

from ..training import XGBSaveFoldsModels

FOLDS_NUM = 10

x_train, x_val = pd.DataFrame(), pd.DataFrame()
y_train, y_val = pd.Series(), pd.Series()
concat_x = pd.concat([x_train, x_val])
concat_y = pd.concat([y_train, y_val])

# Grid Search Parameters
model_gs_params = {
    "xgboost": {
        "max_depth": [3, 4],
        "learning_rate": [0.09, 0.1],
        "early_stopping_rounds": [30],
        'verbosity': [0],
        'n_jobs': [4],
        'random_state': [42]
    },
    "catboost": {
        "n_estimators": [100],
        "max_depth": [3, 4],
        "learning_rate": [0.09, 0.1],
        "early_stopping_rounds": [30],
        "loss_function": ["RMSE"],
        "custom_metric": ["MAE"],
        "use_best_model": [True],
        "verbose": [0],
        'thread_count': [4],
        "random_seed": [42],
    },
}

# CV Parameters
def get_model_cv_params(model_type, params, concat_x, concat_y):
    model_cv_params = {
        "xgboost": {
            "dtrain": DMatrix(concat_x, concat_y),
            "folds": TimeSeriesSplit(n_splits=FOLDS_NUM),
            "params": params,
            "cv_function": xgb_cv,
        },
        "catboost": {
            "pool": Pool(concat_x, concat_y),
            "fold_count": FOLDS_NUM,
            "cv_function": cat_cv,
            "params": params,
        },
    }
    return model_cv_params[model_type]


def get_general_cv_params(model_type, params, cb_instance=None):
    general_cv_params = {
        "xgboost": {
            "num_boost_round": 100,
            "early_stopping_rounds": params["early_stopping_rounds"],
            "metrics": ["rmse", "mae"],
            "as_pandas": True,
            "shuffle": False,
            "callbacks": [cb_instance],
        },
        "catboost": {
            "early_stopping_rounds": params["early_stopping_rounds"],
            "shuffle": False,
            "type": "TimeSeries",
            "return_models": True,
            "plot": False,
            "logging_level": "Silent",
        },
    }
    return general_cv_params[model_type]

# GSCV
## auxiliary functions
def compose_result(model_type, params, cv_results, cb_instance=None):
    if model_type == "xgboost":
        result = {
            "cv_results": cv_results,
            "cv_models": cb_instance.cvboosters,
        }
    if model_type == "catboost":
        result = {"cv_results": cv_results[0], "cv_models": cv_results[1]}
    result["params"] = params
    return result


def run_grid_search(model_type, params, concat_x, concat_y, cb_instance=None):
    model_cv_params = get_model_cv_params(model_type, params, concat_x, concat_y)
    general_cv_params = get_general_cv_params(model_type, params, cb_instance)
    cv_function = model_cv_params.pop("cv_function")
    cv_results = cv_function(**model_cv_params, **general_cv_params)
    result = compose_result(model_type, params, cv_results, cb_instance)
    return result

## run gscv
models_results = {}
for model_type in model_gs_params.keys():
    print(f"Training {model_type}...\n")
    if model_type == "xgboost":
        cb_instance = XGBSaveFoldsModels()
    param_grid = model_gs_params[model_type]
    model_results = Parallel(n_jobs=4, backend="loky")(
        delayed(run_grid_search)(model_type, params, concat_x, concat_y, cb_instance)
        for params in ParameterGrid(param_grid)
    )
    model_results = {str(result["params"]): result for result in model_results}
    models_results[model_type] = model_results

## best results and folds-models for each model type
best_vals = {}
for model_type in model_gs_params.keys():
    best_mae, best_vals[model_type] = np.inf, None
    for result in models_results[model_type].values():
        result["cv_results"].columns = result["cv_results"].columns.str.upper()
        current_mae = result["cv_results"]["TEST-MAE-MEAN"].min()
        if current_mae < best_mae:
            best_mae, best_vals[model_type] = current_mae, result
    print(
        f"Model {model_type} best MAE:",
        best_mae,
        f"with params: {best_vals[model_type]['params']}",
    )
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


def evaluate_baseline(ts, test_indexes, target_col, target_agg_col=None, hhmm_col=None, remove_seasonality=False):
    df = ts.copy()
    df['PREDICTION'] = df[target_col].shift(1)

    if remove_seasonality:
        if target_agg_col is None:
            raise ValueError("`target_agg_col` must be specified when `remove_seasonality` is True")
        df['PREDICTION'] = df['PREDICTION'] + df[target_agg_col]
        df['ORIGINAL'] = df[target_col] + df[target_agg_col]
    else:
        if hhmm_col is None:
            raise ValueError("`hhmm_col` must be specified when `remove_seasonality` is False")
        df.loc[df['hhmm']==940, 'PREDICTION'] = df.loc[df['hhmm']==940, target_col].shift(1)
        df.loc[df['hhmm']==1600, 'PREDICTION'] = df.loc[df['hhmm']==1600, target_col].shift(1)
        df['ORIGINAL'] = df[target_col]

    df_test = df.loc[test_indexes]
    metrics = {}
    metrics['R2'] = r2_score(df_test['ORIGINAL'], df_test['PREDICTION'])
    metrics['MAE'] = mean_absolute_error(df_test['ORIGINAL'], df_test['PREDICTION'])
    metrics['MAPE'] = mean_absolute_percentage_error(df_test['ORIGINAL'], df_test['PREDICTION'])
    return metrics

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
# TODO: change to classes
import numpy as np


def split_data(
    ts,
    test_size=0.1,
    val_size=0.1,
):
    test_len = int(ts.shape[0] * test_size)
    val_len = int(ts.shape[0] * val_size)
    train_len = int(ts.shape[0] - test_len - val_len)

    train_indexes = list(range(train_len))
    val_indexes = list(range(train_len, train_len + val_len))
    test_indexes = list(range(train_len + val_len, ts.shape[0]))
    return train_indexes, val_indexes, test_indexes


def cap_outliers(ts, train_indexes, columns, std_num=4):
    means = ts.loc[train_indexes].mean(numeric_only=True)
    stds = ts.loc[train_indexes].std(numeric_only=True)
    for column in columns:
        up_border = means[column] + std_num * stds[column]
        down_border = max(0, means[column] - std_num * stds[column])
        ts[column] = ts[column].clip(lower=down_border, upper=up_border)
    return ts


def remove_seasonality_ia(ts, columns, base_col="VOLUME_fut", hhmm="hhmm", bins=39, window_days=20):
    # Determine INTRADAY_AVERAGE_VOLUMES, then calculate CURRENT_VOLUME - INTRADAY_AVERAGE_VOLUME
    original_columns = list(ts.columns)
    ts_agg = ts.groupby(by=hhmm).rolling(window_days).mean()
    ts_agg = ts_agg.shift(1).reset_index(level=0).sort_index()

    min_agg_index = bins * window_days + min(ts_agg.index)
    ts_agg[ts_agg.index < min_agg_index] = np.nan

    ts_unseason = ts.join(ts_agg[columns], rsuffix="_agg")
    ts_unseason[columns] = ts[columns] - ts_agg[columns]
    ts_unseason[f"{base_col}_target"] = ts[f"{base_col}_target"] - ts_agg[base_col]

    ts_int_avg = ts_unseason[original_columns + [f"{base_col}_agg"]]

    return ts_int_avg


def restore_seasonality_ia(df_test, base_col="VOLUME_fut"):
    df_test[f"{base_col}_pred"] = (
        df_test[f"{base_col}_pred"] + df_test[f"{base_col}_agg"]
    )
    df_test[f"{base_col}_target"] = (
        df_test[f"{base_col}_target"] + df_test[f"{base_col}_agg"]
    )
    return df_test[[f"{base_col}_pred", f"{base_col}_target"]].dropna()

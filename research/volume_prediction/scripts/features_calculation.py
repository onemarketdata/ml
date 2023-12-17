def add_lags(df, train_indexes, columns, periods=[1, 2, 3, 4, 38, 39, 78, 195]):
    lags_columns = []
    for column in columns:
        for lag in periods:
            feature_col_name = f"{column}_lag_{lag}"
            df[feature_col_name] = df.shift(lag)[column]
            lags_columns.append(feature_col_name)

    df.dropna(inplace=True)
    train_indexes = list(set(df.index) & set(train_indexes))

    return df, lags_columns

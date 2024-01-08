import onetick.py as otp


def load_joined_data(
    etf_ts=None,
    fut_ts=None,
    opt_ts=None,
    common_kwargs={
        "start": otp.dt(2021, 4, 1, 9, 30),
        "end": otp.dt(2022, 4, 1, 16, 0),
        "bucket": 600,
    },
    etf_kwargs={},
    fut_kwargs={},
    opt_kwargs={},
):
    if etf_ts is None:
        etf_ts = _load_etf_data(**common_kwargs, **etf_kwargs)
    if fut_ts is None:
        fut_ts = _load_futures_data(**common_kwargs, **fut_kwargs)
    if opt_ts is None:
        opt_ts = _load_options_data(**common_kwargs, **opt_kwargs)

    ts = fut_ts.join(etf_ts, lsuffix="_fut", rsuffix="_etf", how="inner")
    ts = ts.join(opt_ts, how="inner")
    ts = (
        ts.reset_index()
        .rename(columns={"VOLUME": "VOLUME_opt"})
        .drop(["hhmm_fut", "hhmm_etf"], axis=1)
    )
    ts["VOLUME_fut_target"] = ts["VOLUME_fut"]
    return ts


def _load_etf_data(
    db="NYSE_TAQ",
    tick_type="TRD",
    symbols=["QQQ"],
    start=otp.dt(2021, 4, 1, 9, 30),
    end=otp.dt(2022, 4, 1, 16, 0),
    bucket=600,  # seconds
    timezone="EST5EDT",
):
    data = otp.DataSource(db=db, tick_type=tick_type, symbols=symbols)

    data = data.agg({"VOLUME": otp.agg.sum("SIZE")}, bucket_interval=bucket)
    data, _ = data[data["VOLUME"] > 0]

    data["hhmm"] = data["Time"].dt.strftime(format="%H%M")
    data["hhmm"] = data["hhmm"].apply(int)

    etf_ts = otp.run(
        data, start=start, end=end, apply_times_daily=True, timezone=timezone
    ).set_index("Time")
    return etf_ts


def _load_futures_data(
    db="CME",
    tick_type="TRD",
    symbols=["NQ\H21", "NQ\M21", r"NQ\U21", "NQ\Z21", "NQ\H22", "NQ\M22"],  # H M U Z
    start=otp.dt(2021, 4, 1, 9, 30),
    end=otp.dt(2022, 4, 1, 16, 0),
    bucket=600,
    timezone="EST5EDT",
):
    data = otp.DataSource(db=db, tick_type=tick_type, symbols=symbols)

    data = data.agg({"VOLUME": otp.agg.sum("SIZE")}, bucket_interval=bucket)
    data, _ = data[data["VOLUME"] > 0]

    data["hhmm"] = data["Time"].dt.strftime(format="%H%M")
    data["hhmm"] = data["hhmm"].apply(int)

    fut_ts = otp.run(
        data, start=start, end=end, apply_times_daily=True, timezone=timezone
    ).set_index("Time")
    return fut_ts


def _load_options_data(
    db="US_OPTIONS",
    tick_type="TRD",
    start=otp.dt(2021, 4, 1),
    end=otp.dt(2022, 4, 1),
    bucket=600,
    timezone="EST5EDT",
):
    data = otp.DataSource(db=db, tick_type=tick_type, identify_input_ts=True)
    data["Date"] = data["Time"].dt.date()

    volume_date = data.agg({"VOLUME": otp.agg.sum("SIZE")}, group_by=["Date"])
    volume_10min = data.agg({"VOLUME": otp.agg.sum("SIZE")}, bucket_interval=bucket)

    all_symbols = otp.Symbols(
        db="US_OPTIONS", start=start, end=end, keep_db=True, pattern="QQQ%"
    )

    symbols_date = otp.merge([volume_date], symbols=all_symbols, identify_input_ts=True)

    most_traded_by_days = symbols_date.high("VOLUME", n=5, group_by=["Date"])
    most_traded_by_days = otp.run(
        most_traded_by_days, start=start, end=end, timezone=timezone
    )

    symbols = list(pd.unique(most_traded_by_days["SYMBOL_NAME"]))
    symbols_10min = otp.merge([volume_10min], symbols=symbols)

    symbols_10min["hhmm"] = symbols_10min["Time"].dt.strftime(format="%H%M")
    symbols_10min["hhmm"] = symbols_10min["hhmm"].apply(int)
    symbols_10min["_Time"] = symbols_10min["Time"]

    symbols_10min = symbols_10min.agg(
        {
            "VOLUME": otp.agg.sum("VOLUME"),
            "_Time": otp.agg.first("_Time"),
            "hhmm": otp.agg.first("hhmm"),
        },
        group_by="_Time",
    )

    symbols_10min["Time"] = symbols_10min["_Time"]
    symbols_10min = symbols_10min.drop("_Time")
    symbols_10min, _ = symbols_10min[symbols_10min["VOLUME"] > 0]
    symbols_10min, _ = symbols_10min[
        (symbols_10min["hhmm"] > 930) & (symbols_10min["hhmm"] <= 1600)
    ]

    opt_ts = otp.run(symbols_10min, start=start, end=end, timezone=timezone).set_index(
        "Time"
    )
    return opt_ts

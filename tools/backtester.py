# import pandas as pd
# import pyfolio
# import matplotlib
#
# df=pd.read_csv('data/dow_30_2009_2020.csv')
# rebalance_window = 63
# validation_window = 63
# unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()
# df_trade_date = pd.DataFrame({'datadate':unique_trade_date})
# ensemble_account_value = get_account_value('ensemble')
# ensemble_account_value.account_value.plot()
# ensemble_account_value = get_daily_return(ensemble_account_value)
# ensemble_strat = backtest_strat(ensemble_account_value[0:1097])
#
# with pyfolio.plotting.plotting_context(font_scale=1.1):
#     pyfolio.create_full_tear_sheet(returns = ensemble_strat,
#                                    benchmark_rets=dow_strat, set_context=False)
#
# def get_daily_return(df):
#     df['daily_return']=df.account_value.pct_change(1)
#     #df=df.dropna()
#     print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
#     return df
#
#
# def backtest_strat(df):
#     strategy_ret= df.copy()
#     strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
#     strategy_ret.set_index('Date', drop = False, inplace = True)
#     strategy_ret.index = strategy_ret.index.tz_localize('UTC')
#     del strategy_ret['Date']
#     ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
#     return ts
#
#
#
# def get_account_value(model_name):
#     df_account_value=pd.DataFrame()
#     for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
#         temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format(model_name,i))
#         df_account_value = df_account_value.append(temp,ignore_index=True)
#     df_account_value = pd.DataFrame({'account_value':df_account_value['0']})
#     sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
#     print(sharpe)
#     df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
#     return df_account_value
#
#
# def dija():
#     dji = pd.read_csv("data/^DJI.csv")
#     test_dji = dji[(dji['Date'] >= '2016-01-01') & (dji['Date'] <= '2020-06-30')]
#     test_dji = test_dji.reset_index(drop=True)
#     test_dji['daily_return']=test_dji['Adj Close'].pct_change(1)
#     dow_strat = backtest_strat(test_dji)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import pyfolio.timeseries as ts

# ------- helper：把各种类型（int位置/字符串/时间戳/NaT）统一成 Timestamp/NaT -------
def _to_timestamp_or_nat(x, idx):
    if x is None or pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, np.integer)):         # 有些版本会返回位置索引
        return pd.Timestamp(idx[int(x)])
    return pd.Timestamp(x)

# ------- 1) patch get_top_drawdowns：保证用于绘图的 recovery 一定是一个日期 -------
_orig_get_top_drawdowns = ts.get_top_drawdowns

def get_top_drawdowns_fixed(returns, top=10):
    returns = returns.dropna()
    idx = returns.index
    last_date = idx[-1]

    dds = _orig_get_top_drawdowns(returns, top=top)
    fixed = []
    for peak, valley, recovery in dds:
        peak_ts = _to_timestamp_or_nat(peak, idx)
        valley_ts = _to_timestamp_or_nat(valley, idx)
        recovery_ts = _to_timestamp_or_nat(recovery, idx)

        # 关键：若没恢复，用样本末尾日期代替（用于绘图）
        if pd.isna(recovery_ts):
            recovery_ts = last_date

        fixed.append((peak_ts, valley_ts, recovery_ts))
    return fixed

ts.get_top_drawdowns = get_top_drawdowns_fixed

# ------- 2) patch gen_drawdown_table：不再 strftime(NaT)，并增加 Recovered 标记 -------
_orig_gen_drawdown_table = ts.gen_drawdown_table

def gen_drawdown_table_fixed(returns, top=10):
    returns = returns.dropna()
    idx = returns.index
    last_date = idx[-1]

    # underwater 曲线用于计算回撤幅度
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    underwater = cum / running_max - 1.0

    # 用“原始”top drawdowns 来判断是否真实恢复（recovery 是否缺失）
    dds_raw = _orig_get_top_drawdowns(returns, top=top)

    rows = []
    for peak, valley, recovery in dds_raw:
        peak_ts = _to_timestamp_or_nat(peak, idx)
        valley_ts = _to_timestamp_or_nat(valley, idx)
        recovery_ts = _to_timestamp_or_nat(recovery, idx)

        recovered = not pd.isna(recovery_ts)
        recovery_plot = recovery_ts if recovered else last_date  # 画图/持续时间用

        net_dd = float(underwater.loc[valley_ts]) * 100.0 if pd.notna(valley_ts) else np.nan
        duration = (recovery_plot - peak_ts).days if (pd.notna(peak_ts) and pd.notna(recovery_plot)) else np.nan

        rows.append({
            "Peak date": peak_ts,
            "Valley date": valley_ts,
            "Recovery date": recovery_plot,   # 没恢复则=样本末尾
            "Recovered": recovered,           # 关键标记：是否真实恢复
            "Duration": duration,
            "Net drawdown in %": net_dd,
        })

    return pd.DataFrame(rows)

ts.gen_drawdown_table = gen_drawdown_table_fixed
def get_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_return and print Sharpe."""
    out = df.copy()
    out["daily_return"] = out["account_value"].pct_change(1)
    dr = out["daily_return"].dropna()
    if dr.std(ddof=0) != 0:
        sharpe = np.sqrt(252) * dr.mean() / dr.std(ddof=0)
    else:
        sharpe = np.nan
    print("Sharpe:", sharpe)
    return out


def backtest_strat(df: pd.DataFrame) -> pd.Series:
    """
    Convert a df containing Date(+daily_return) or datadate(+daily_return)
    into a UTC-indexed daily return Series for pyfolio.
    """
    s = df.copy()

    # unify date column
    if "Date" not in s.columns:
        if "datadate" in s.columns:
            # datadate is usually like 20160104
            s["Date"] = pd.to_datetime(s["datadate"]).dt.strftime("%Y%m%d")
            #s["Date"] = pd.to_datetime(s["datadate"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            raise ValueError("backtest_strat(): need column 'Date' or 'datadate'.")

    s["Date"] = pd.to_datetime(s["Date"])
    s.set_index('Date', drop=True, inplace=True)
    if isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.tz_localize('UTC')
    else:
        raise ValueError("Index is not a DatetimeIndex, cannot localize timezone.")
    ts = pd.Series(s["daily_return"].values, index=s.index, name="daily_return").dropna()
    return ts


# -----------------------------
# Data loaders
# -----------------------------
def get_account_value(
    model_name: str,
    results_dir: str,
    unique_trade_date: np.ndarray,
    df_trade_date: pd.DataFrame,
    rebalance_window: int,
    validation_window: int,
) -> pd.DataFrame:
    """
    Load account value csv fragments and stitch into one DataFrame:
    columns: account_value, Date
    """
    dfs = []
    for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
        fp = os.path.join(results_dir, f"account_value_trade_{model_name}_{i}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")
        dfs.append(pd.read_csv(fp))

    df_all = pd.concat(dfs, ignore_index=True)

    # robustly find the account value column
    if "account_value" in df_all.columns:
        av = df_all["account_value"]
    elif "0" in df_all.columns:
        av = df_all["0"]
    else:
        av = df_all.iloc[:, 0]  # fallback to first column

    out = pd.DataFrame({"account_value": av.astype(float)})

    # attach trade dates (skip validation window)
    trade_dates = df_trade_date[validation_window:].reset_index(drop=True)

    # length alignment safety
    if len(trade_dates) != len(out):
        min_len = min(len(trade_dates), len(out))
        out = out.iloc[:min_len].reset_index(drop=True)
        trade_dates = trade_dates.iloc[:min_len].reset_index(drop=True)

    out = out.join(trade_dates)

    # rename and parse date
    out = out.rename(columns={"datadate": "Date"})
    out["Date"] = pd.to_datetime(out['Date']).dt.strftime("%Y%m%d")
    return out


def get_dow_benchmark(dji_path="../data/^CSI300.csv", start="2023-12-26", end="2026-01-27") -> pd.Series:
    """Load DJI benchmark and return UTC-indexed daily return Series."""
    dji = pd.read_csv(dji_path)
    test = dji[(dji["Date"] >= start) & (dji["Date"] <= end)].copy()
    test = test.reset_index(drop=True)
    test["daily_return"] = test["Close"].pct_change(1)
    return backtest_strat(test)


# -----------------------------
# Main
# -----------------------------
def main():
    # df = pd.read_csv("data/dow_30_2009_2020.csv")
    df = pd.read_pickle(f'../output/predictions/002905_predictions.pkl')
    df['Date'] = pd.to_datetime(df['Date'])
    rebalance_window = 63
    validation_window = 20

    unique_trade_date = df['Date'].unique().tolist()
    df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

    # 1) strategy account value
    ensemble_account_value = get_account_value(
        model_name="ppo",
        results_dir="../output/results",
        unique_trade_date=unique_trade_date,
        df_trade_date=df_trade_date,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
    )

    # plot account value
    ensemble_account_value.set_index("Date")["account_value"].plot(title="Ensemble Account Value")
    plt.show()

    # 2) convert to daily return series
    ensemble_account_value = get_daily_return(ensemble_account_value)
    ensemble_strat = backtest_strat(ensemble_account_value)

    # 3) benchmark
    # csi_df = ak.index_zh_a_hist(symbol="000300", period="daily")
    # column_mapping = {
    #     '日期': 'Date',
    #     '开盘': 'Open',
    #     '收盘': 'Close',
    #     '最高': 'High',
    #     '最低': 'Low',
    #     '成交量': 'Volume',
    #     '成交额': 'Turnover'
    # }
    #
    # # 重命名列名
    # csi_df.rename(columns=column_mapping, inplace=True)
    # selected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # df_selected = csi_df[selected_columns]
    # df_selected.to_csv("../data/^CSI300.csv", index=False)
    dow_strat = get_dow_benchmark()

    # 4) align indices (important for pyfolio)
    common_index = ensemble_strat.index.intersection(dow_strat.index)
    ensemble_strat = ensemble_strat.loc[common_index]
    dow_strat = dow_strat.loc[common_index]

    print(type(ensemble_strat.index), ensemble_strat.index.dtype, ensemble_strat.index[:3])
    print(type(dow_strat.index), dow_strat.index.dtype, dow_strat.index[:3])
    # 5) tear sheet
    with pf.plotting.plotting_context(font_scale=1.1):
        pf.create_full_tear_sheet(
            returns=ensemble_strat,
            benchmark_rets=dow_strat,
            set_context=False,
        )

    plt.show()
if __name__ == "__main__":
    main()
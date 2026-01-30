import asyncio, functools
import akshare as ak
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from services.drlPortfolioService import run_drl_portfolio
from services.lstmProcess import lstm_stock_predict
from services.technicalService import get_history_data


# 同步函数：保持不变
# def get_history_data(ticker, start_date, end_date): ...
async def fetch_all(stock_codes, start_date, end_date, max_workers=8):
    loop = asyncio.get_running_loop()
    def run_one(code):
        # 这里可做统一异常捕获
        return code, get_history_data(code, start_date, end_date)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, functools.partial(run_one, code))
            for code in stock_codes
        ]
        # 并发执行，同步函数在线程池里跑
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # 整理为 {code: df or Exception}
    results = {}
    for code, item in zip(stock_codes, results_list):
        if isinstance(item, Exception):
            # 记录失败原因，别让单个失败拖垮整体
            results[code] = item
        else:
            results[code] = item  # 正常的 DataFrame
    return results

if __name__ == '__main__':
    # results = asyncio.run(fetch_all(['600518', '600519', '600598', '600522', '600487'], '2025-12-01', '2026-01-25'))
    # results = get_history_data('600518', '2025-12-01', '2026-01-25')
    # print(results)
    pre_list = ['002905', '600795', '300442']
    hold = {'stocks': {'002905': [9.3, 1000], '600795': [3.7, 2000], '300442': [23.33, 1000]}, 'capital': 1000000}
    features = [
        'Tic', 'Open', 'Close', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR', 'predict_percentages',
        'prediction'
    ]
    # 读取并找出最晚起始日期
    all_dfs = []
    max_start_date = None

    for ticker in pre_list:
        df = pd.read_pickle(f'./output/predictions/{ticker}_predictions.pkl')
        df['Tic'] = ticker
        df = df[['Date'] + features]
        df['Date'] = pd.to_datetime(df['Date'])
        # 为除Date外的所有列添加股票代码后缀
        cols_to_rename = {}
        for col in df.columns:
            if col != 'Date':  # 保留Date和Tic列名不变
                cols_to_rename[col] = f"{col}_{ticker}"

        df = df.rename(columns=cols_to_rename)
        current_start_date = df['Date'].min()
        if max_start_date is None or current_start_date > max_start_date:
            max_start_date = current_start_date

        all_dfs.append(df)

        # 从统一起始日期截取并拼接
    stock_dfs = []
    for df in all_dfs:
        df_filtered = df[df['Date'] >= max_start_date].copy()
        stock_dfs.append(df_filtered)

    # 横向拼接：按Date列合并
    stock_df = stock_dfs[0]  # 以第一只股票的数据为基准
    for df in stock_dfs[1:]:
        stock_df = pd.merge(stock_df, df, on='Date', how='outer')
    unique_trade_date = stock_df['Date'].unique().tolist()
    rebalance = 63  # 季度再平衡 (约3个月)
    validation = 20  # 验证期
    run_drl_portfolio(
        stock_df=stock_df,
        unique_trade_date=unique_trade_date,
        rebalance=rebalance,
        validation=validation,
        hold=hold
    )



    # 用法
    # results = asyncio.run(fetch_all(stock_code, start_date, end_date, max_workers=8))

import asyncio, functools
import akshare as ak
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from matplotlib import pyplot as plt

from models.gflownets import run_gflownets
from services.drlPortfolioService import run_drl_portfolio
from services.lstmProcess import lstm_stock_predict, async_lstm_all_stock
from services.stockDataService import StockDataService
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
    # , '601628', '688036'
    # asyncio.run(async_lstm_all_stock(['sh.601319']))
    pre_list = ['000776', '002600', '002905', '600795', '601877', '002001', '300628', '000338', '601628', '601336']
    hold = {'stocks': {'000776': [11.98, 3000], '002600': [13.57, 2000], '002905': [9.3, 3000], '600795': [4.713, 6800], '601877': [5.07, 3000], '002001': [0, 0], '300628': [0, 0], '000338': [0, 0], '601628': [0, 0], '601336': [0, 0]}, 'capital': 1000000}
    # pre_list = ['000776', '002600', '002905', '600795', '601877', '002001', '300442', '601319', '601628', '601336']
    # hold = {'stocks': {'000776': [11.98, 3000], '002600': [13.57, 2000], '002905': [9.3, 3000], '600795': [4.713, 6800],
    #                    '601877': [5.07, 3000], '002001': [0, 0], '300442': [0, 0], '601319': [0, 0], '601628': [0, 0],
    #                    '601336': [0, 0]}, 'capital': 1000000}
    features = [
        'Tic', 'Open', 'Close', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Relative_Performance', 'ATR', 'predict_percentages',
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
    start_date = pd.to_datetime('2024-04-01')
    end_date = pd.to_datetime('2026-01-26')
    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)].copy()
    prediction_columns = [col for col in stock_df.columns if 'prediction' in col]
    prediction_data = stock_df[prediction_columns]
    unique_trade_date = pd.to_datetime(stock_df['Date'].unique()).strftime('%Y-%m-%d').tolist()
    rebalance = 63  # 季度再平衡 (约3个月)
    validation = 20  # 验证期
    run_drl_portfolio(
        stock_df=stock_df,
        unique_trade_date=unique_trade_date,
        rebalance=rebalance,
        validation=validation,
        hold=hold
    )

    #
    # strength_scores = [74, 73, 72, 72, 70, 65, 65, 62, 61, 59]
    # risk_scores = [6.05, 8.35, 6.55, 8.05, 7.2, 7.55, 8.05, 8.05, 8.2, 7.95]
    # run_gflownets(strength_scores, risk_scores)
    # 用法
    # results = asyncio.run(fetch_all(stock_code, start_date, end_date, max_workers=8))


    """
    previous_total_asset: 1000000
    end_total_asset: 1161991.1404204401
    total_reward: 161991.14042044012
    total_cost: 574.7826747990472
    total
    trades: 364
    Sharpe: 2.0084293731093674
    
    previous_total_asset:1161991.1404204401
end_total_asset:1477175.815107437
total_reward:315184.67468699696
total_cost:  718.0259078575549
total trades:  378
Sharpe:  3.918142077944271

    previous_total_asset:1477175.815107437
end_total_asset:1309420.1561353656
total_reward:-167755.65897207148
total_cost:  736.5383346172118
total trades:  496
Sharpe:  -2.2022420218551177

    previous_total_asset:1309420.1561353656
end_total_asset:1363034.7566238092
total_reward:53614.600488443626
total_cost:  938.5658640256262
total trades:  436
Sharpe:  0.789261150436623

    previous_total_asset:1363034.7566238092
end_total_asset:1528165.594436425
total_reward:165130.8378126158
total_cost:  1192.8283959571786
total trades:  494
Sharpe:  4.722173212086993

    previous_total_asset:1528165.594436425
end_total_asset:1622701.6421586978
total_reward:94536.04772227281
total_cost:  1070.1048672971474
total trades:  496
Sharpe:  1.5512131707704304


-------------------------------------------------ppo
previous_total_asset:1000000
end_total_asset:1160607.9480565947
total_reward:160607.9480565947
total_cost:  597.7552445005917
total trades:  589
Sharpe:  2.0023210905530977

previous_total_asset:1160607.9480565947
end_total_asset:1222121.8034248783
total_reward:61513.85536828358
total_cost:  738.5367687075744
total trades:  609
Sharpe:  2.6383425831070695

previous_total_asset:1222121.8034248783
end_total_asset:1211659.5741227213
total_reward:-10462.22930215695
total_cost:  927.4600934059712
total trades:  620
Sharpe:  -0.49191201672262874
previous_total_asset:1211659.5741227213
end_total_asset:1199988.7122483887
total_reward:-11670.861874332651
total_cost:  994.094973307068
total trades:  619
Sharpe:  -0.33213629301088177

previous_total_asset:1199988.7122483887
end_total_asset:1273454.2798485975
total_reward:73465.5676002088
total_cost:  1009.722653040062
total trades:  614
Sharpe:  4.626748406116038

previous_total_asset:1273454.2798485975
end_total_asset:1314709.0016383645
total_reward:41254.721789767034
total_cost:  1154.9705154329542
total trades:  620
Sharpe:  1.299788806335036

------------------------------------ppo2
previous_total_asset:1000000
end_total_asset:1162652.80145754
total_reward:162652.8014575399
total_cost:  668.4025154370979
total trades:  593
Sharpe:  2.024088903325796
previous_total_asset:1162652.80145754
end_total_asset:1224170.7592355148
total_reward:61517.957777974894
total_cost:  698.2976872794111
total trades:  595
Sharpe:  2.8372931625815236
previous_total_asset:1224170.7592355148
end_total_asset:1196239.7468945389
total_reward:-27931.012340975925
total_cost:  825.6134910553488
total trades:  608
Sharpe:  -1.8726951518243165
previous_total_asset:1196239.7468945389
end_total_asset:1206334.9295769439
total_reward:10095.182682404993
total_cost:  881.1852260636882
total trades:  619
Sharpe:  0.6374548732132946
previous_total_asset:1206334.9295769439
end_total_asset:1240623.026415881
total_reward:34288.09683893714
total_cost:  920.8884418682273
total trades:  614
Sharpe:  3.792837162686854
previous_total_asset:1240623.026415881
end_total_asset:1280796.2431627777
total_reward:40173.21674689674
total_cost:  1088.5045089584191
total trades:  614
Sharpe:  1.9198632216027693
---------------------sac2  
previous_total_asset:1000000
end_total_asset:1169885.7664602152
total_reward:169885.7664602152
total_cost:  514.8295670306176
total trades:  378
Sharpe:  2.092443065070023

previous_total_asset:1169885.7664602152
end_total_asset:1283129.502789368
total_reward:113243.73632915271
total_cost:  713.0017974209046
total trades:  440
Sharpe:  3.195457977368351

previous_total_asset:1283129.502789368
end_total_asset:1250813.7357961687
total_reward:-32315.766993199242
total_cost:  1319.518297131002
total trades:  578
Sharpe:  -0.5683193859539596
previous_total_asset:1250813.7357961687
end_total_asset:1247109.9604746825
total_reward:-3703.775321486173
total_cost:  371.06745137326084
total trades:  449
Sharpe:  0.07046513906641434
previous_total_asset:1247109.9604746825
end_total_asset:1406512.0755365267
total_reward:159402.11506184423
total_cost:  1253.0473458161916
total trades:  410
Sharpe:  4.133979268703287
previous_total_asset:1406512.0755365267
end_total_asset:1523677.9416054108
total_reward:117165.86606888403
total_cost:  720.9444984073376
total trades:  372
Sharpe:  2.108305826157972
------------------------------------------sac3
previous_total_asset:1000000
end_total_asset:1150512.5691450383
total_reward:150512.56914503826
total_cost:  732.0790419094549
total trades:  484
Sharpe:  1.8635756036240083

previous_total_asset:1150512.5691450383
end_total_asset:1634115.863330332
total_reward:483603.29418529384
total_cost:  645.9382753008123
total trades:  558
Sharpe:  3.620466580217832

previous_total_asset:1634115.863330332
end_total_asset:1603860.1006269604
total_reward:-30255.7627033717
total_cost:  778.9181027374614
total trades:  591
Sharpe:  -0.17600349525010517

previous_total_asset:1603860.1006269604
end_total_asset:1601215.672851518
total_reward:-2644.4277754423674
total_cost:  1415.6626082513026
total trades:  558
Sharpe:  0.11291670031686367

previous_total_asset:1601215.672851518
end_total_asset:1697963.818361932
total_reward:96748.14551041392
total_cost:  1273.5293887348082
total trades:  543
Sharpe:  2.236752221663688

previous_total_asset:1697963.818361932
end_total_asset:1782523.9575680632
total_reward:84560.13920613122
total_cost:  994.3001191032162
total trades:  430
Sharpe:  1.52467627137148
--------------------------------------  ppo3
previous_total_asset:1000000
end_total_asset:1158454.721986373
total_reward:158454.72198637296
total_cost:  580.0581711612092
total trades:  587
Sharpe:  1.978621311691866

previous_total_asset:1158454.721986373
end_total_asset:1231588.4523072313
total_reward:73133.7303208583
total_cost:  734.050415867437
total trades:  618
Sharpe:  3.287108818963211

previous_total_asset:1231588.4523072313
end_total_asset:1209885.2484717763
total_reward:-21703.203835455002
total_cost:  889.5467577914333
total trades:  617
Sharpe:  -1.4474610565337562

previous_total_asset:1209885.2484717763
end_total_asset:1222182.9132230668
total_reward:12297.664751290577
total_cost:  940.5196980016343
total trades:  602
Sharpe:  0.6361812639777332

previous_total_asset:1222182.9132230668
end_total_asset:1277142.9197745647
total_reward:54960.00655149785
total_cost:  879.3159345368957
total trades:  597
Sharpe:  4.148057541378988

previous_total_asset:1277142.9197745647
end_total_asset:1306589.1447685352
total_reward:29446.224993970478
total_cost:  1055.383820484977
total trades:  610
Sharpe:  1.466943844520228
    """


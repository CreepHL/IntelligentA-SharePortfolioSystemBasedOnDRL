import asyncio, functools
import akshare as ak
from concurrent.futures import ThreadPoolExecutor

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
    asyncio.run(lstm_stock_predict('002600'))



    # 用法
    # results = asyncio.run(fetch_all(stock_code, start_date, end_date, max_workers=8))

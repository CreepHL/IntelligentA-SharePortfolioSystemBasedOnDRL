from datetime import datetime, timedelta
import pandas as pd
import akshare as ak
from tools.logConfig import log_config

logger = log_config('api')


def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    计算股票的技术指标

    参数:
        data: DataFrame, 包含OHLCV数据的DataFrame
        start_date: str, 开始日期 (可选，用于相对表现计算)
        end_date: str, 结束日期 (可选，用于相对表现计算)

    返回:
        DataFrame: 添加了技术指标的数据
    """
    # 添加日期特征
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # 移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()

    # RSI指标
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD指标
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    # VWAP指标
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # 布林带
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘表现
    if start_date and end_date:
        # benchmark_data = ak.stock_zh_index_daily_em(symbol="csi931151")  # 筛选日期范围
        # benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
        # mask = (benchmark_data['date'] >= start_date) & (benchmark_data['date'] <= end_date)
        # benchmark_subset = benchmark_data.loc[mask, 'close']
        # if not benchmark_subset.empty:
        #     data['Relative_Performance'] = (data['Close'] / benchmark_subset.values) * 100
        # else:
        data['Relative_Performance'] = 24.05

    # ROC指标
    data['ROC'] = data['Close'].pct_change(periods=1) * 100

    # ATR指标
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)

    # 删除缺失值
    data = data.dropna()

    return data


def get_stock_data(ticker, start_date, end_date):
    """
    获取并处理单个股票的数据

    参数:
        ticker: 股票代码
        start_date: 起始日期
        end_date: 结束日期
    返回:
        处理后的股票数据DataFrame
    """
    # 下载股票数据
    # data = yf.download(ticker, start=start_date, end=end_date)  # 无代理
    # data = yf.download(ticker, start=start_date, end=end_date, proxy="http://127.0.0.1:7890")  # 有代理
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    df = ak.stock_zh_a_hist(
        symbol=ticker,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust='qfq'
    )

    # 重命名列以匹配技术分析代理的需求
    df = df.rename(columns={
        "日期": "Date",
        "开盘": "Open",
        "最高": "High",
        "最低": "Low",
        "收盘": "Close",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change_amount",
        "换手率": "turnover"
    })
    # 确保日期列为datetime类型
    df["Date"] = pd.to_datetime(df["Date"])
    # 计算技术指标
    data = calculate_technical_indicators(df, start_date, end_date)

    return data
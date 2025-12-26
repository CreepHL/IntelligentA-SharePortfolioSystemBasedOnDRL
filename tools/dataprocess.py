import pandas as pd
import numpy as np
from pydantic import json
from sklearn.preprocessing import MinMaxScaler

class DataProcess:

    @staticmethod
    def preprocess_data(data):
        df = data.copy()
        df.index = pd.to_datetime(df.index)

        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_limit, lower_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            df[col] = df[col].clip(lower_limit, upper_limit)

        original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Pct_Change'] = df['Close'].pct_change() * 100
        df_display = df[original_columns + ['Pct_Change']]

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
            df[['Open', 'High', 'Low', 'Close', 'Volume']])
        return df, scaler, df_display

    @staticmethod  # Make it a static method
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data

    @staticmethod  # Make it a static method
    def calculate_macd(data):
        short_ema = data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = short_ema - long_ema
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data

    @staticmethod
    def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
        sma = prices_df['close'].rolling(window).mean()
        std_dev = prices_df['close'].rolling(window).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        return upper_band, lower_band

    @staticmethod
    def SMA(data):
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data

    @staticmethod
    def EMA(data):
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        return data

    @staticmethod
    def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
        obv = [0]
        for i in range(1, len(prices_df)):
            if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
                obv.append(obv[-1] + prices_df['volume'].iloc[i])
            elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
                obv.append(obv[-1] - prices_df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        prices_df['OBV'] = obv
        return prices_df['OBV']

    @staticmethod  # Make it a static method
    def recommend_stock_action(data):
        data = DataProcess.calculate_macd(data)
        data = DataProcess.calculate_rsi(data)
        data = DataProcess.SMA(data)

        data['SMA_200'] = data['Close'].rolling(window=200).mean()

        latest_rsi = data['RSI'].iloc[-1]
        latest_macd = data['MACD'].iloc[-1]
        latest_signal = data['Signal_Line'].iloc[-1]
        latest_sma50 = data['SMA_50'].iloc[-1]
        latest_sma200 = data['SMA_200'].iloc[-1]
        latest_ema50 = data['EMA_50'].iloc[-1]
        latest_ema200 = data['EMA_200'].iloc[-1]

        buy_signals, sell_signals = [], []

        if latest_sma50 > latest_sma200:
            buy_signals.append("SMA_50 above SMA_200 (Golden Cross)")
        elif latest_sma50 < latest_sma200:
            sell_signals.append("SMA_50 below SMA_200 (Death Cross)")

        if latest_ema50 > latest_ema200:
            buy_signals.append("EMA_50 above EMA_200 (Golden Cross)")
        elif latest_ema50 < latest_ema200:
            sell_signals.append("EMA_50 below EMA_200 (Death Cross)")

        if latest_rsi < 30:
            buy_signals.append("RSI below 30 (Oversold)")
        elif latest_rsi > 70:
            sell_signals.append("RSI above 70 (Overbought)")

        if latest_macd > latest_signal:
            buy_signals.append("MACD above Signal Line")
        elif latest_macd < latest_signal:
            sell_signals.append("MACD below Signal Line")

        if len(buy_signals) > len(sell_signals):
            return f"**Recommendation: BUY** 📈/nReasons: {', '.join(buy_signals)}"
        elif len(sell_signals) > len(buy_signals):
            return f"**Recommendation: SELL** 📉/nReasons: {', '.join(sell_signals)}"
        else:
            return "**Recommendation: HOLD** 🤔/nMarket is neutral or mixed signals."

    @staticmethod
    def technical_analysis(data):
        prices = data["prices"]
        # prices_df = prices_to_df(prices)

        # Initialize confidence variable
        confidence = 0.0

        # Calculate indicators
        # 1. MACD (Moving Average Convergence Divergence)
        macd_line, signal_line = DataProcess.calculate_macd(prices)

        # 2. RSI (Relative Strength Index)
        rsi = DataProcess.calculate_rsi(prices)

        # 3. Bollinger Bands (Bollinger Bands)
        upper_band, lower_band = DataProcess.calculate_bollinger_bands(prices)

        # 4. OBV (On-Balance Volume)
        obv = DataProcess.calculate_obv(prices)

        # Generate individual signals
        signals = []

        # MACD signal
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals.append('bullish')
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            signals.append('bearish')
        else:
            signals.append('neutral')

        # RSI signal
        if rsi.iloc[-1] < 30:
            signals.append('bullish')
        elif rsi.iloc[-1] > 70:
            signals.append('bearish')
        else:
            signals.append('neutral')

        # Bollinger Bands signal
        current_price = prices['close'].iloc[-1]
        if current_price < lower_band.iloc[-1]:
            signals.append('bullish')
        elif current_price > upper_band.iloc[-1]:
            signals.append('bearish')
        else:
            signals.append('neutral')

        # OBV signal
        obv_slope = obv.diff().iloc[-5:].mean()
        if obv_slope > 0:
            signals.append('bullish')
        elif obv_slope < 0:
            signals.append('bearish')
        else:
            signals.append('neutral')

        # Calculate price drop
        price_drop = (prices['close'].iloc[-1] - prices['close'].iloc[-5]) / prices['close'].iloc[-5]

        # Add price drop signal
        if price_drop < -0.05 and rsi.iloc[-1] < 40:  # 5% drop and RSI below 40
            signals.append('bullish')
            confidence += 0.2  # Increase confidence for oversold conditions
        elif price_drop < -0.03 and rsi.iloc[-1] < 45:  # 3% drop and RSI below 45
            signals.append('bullish')
            confidence += 0.1

        # Add reasoning collection
        reasoning = {
            "MACD": {
                "signal": signals[0],
                "details": f"MACD Line crossed {'above' if signals[0] == 'bullish' else 'below' if signals[0] == 'bearish' else 'neither above nor below'} Signal Line"
            },
            "RSI": {
                "signal": signals[1],
                "details": f"RSI is {rsi.iloc[-1]:.2f} ({'oversold' if signals[1] == 'bullish' else 'overbought' if signals[1] == 'bearish' else 'neutral'})"
            },
            "Bollinger": {
                "signal": signals[2],
                "details": f"Price is {'below lower band' if signals[2] == 'bullish' else 'above upper band' if signals[2] == 'bearish' else 'within bands'}"
            },
            "OBV": {
                "signal": signals[3],
                "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})"
            }
        }

        # Determine overall signal
        bullish_signals = signals.count('bullish')
        bearish_signals = signals.count('bearish')

        if bullish_signals > bearish_signals:
            overall_signal = 'bullish'
        elif bearish_signals > bullish_signals:
            overall_signal = 'bearish'
        else:
            overall_signal = 'neutral'

        # Calculate confidence level based on the proportion of indicators agreeing
        total_signals = len(signals)
        confidence = max(bullish_signals, bearish_signals) / total_signals

        return total_signals

    @staticmethod
    def fundermental_analysis(data):
        metrics = data["financial_metrics"][0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        # 1. Profitability Analysis
        return_on_equity = metrics.get("return_on_equity", 0)
        net_margin = metrics.get("net_margin", 0)
        operating_margin = metrics.get("operating_margin", 0)

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15)  # Strong operating efficiency
        ]
        profitability_score = sum(
            metric is not None and metric > threshold
            for metric, threshold in thresholds
        )

        signals.append('bullish' if profitability_score >=
                                    2 else 'bearish' if profitability_score == 0 else 'neutral')
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (
                           f"ROE: {metrics.get('return_on_equity', 0):.2%}" if metrics.get(
                               "return_on_equity") is not None else "ROE: N/A"
                       ) + ", " + (
                           f"Net Margin: {metrics.get('net_margin', 0):.2%}" if metrics.get(
                               "net_margin") is not None else "Net Margin: N/A"
                       ) + ", " + (
                           f"Op Margin: {metrics.get('operating_margin', 0):.2%}" if metrics.get(
                               "operating_margin") is not None else "Op Margin: N/A"
                       )
        }

        # 2. Growth Analysis
        revenue_growth = metrics.get("revenue_growth", 0)
        earnings_growth = metrics.get("earnings_growth", 0)
        book_value_growth = metrics.get("book_value_growth", 0)

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10)  # 10% book value growth
        ]
        growth_score = sum(
            metric is not None and metric > threshold
            for metric, threshold in thresholds
        )

        signals.append('bullish' if growth_score >=
                                    2 else 'bearish' if growth_score == 0 else 'neutral')
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (
                           f"Revenue Growth: {metrics.get('revenue_growth', 0):.2%}" if metrics.get(
                               "revenue_growth") is not None else "Revenue Growth: N/A"
                       ) + ", " + (
                           f"Earnings Growth: {metrics.get('earnings_growth', 0):.2%}" if metrics.get(
                               "earnings_growth") is not None else "Earnings Growth: N/A"
                       )
        }

        # 3. Financial Health
        current_ratio = metrics.get("current_ratio", 0)
        debt_to_equity = metrics.get("debt_to_equity", 0)
        free_cash_flow_per_share = metrics.get("free_cash_flow_per_share", 0)
        earnings_per_share = metrics.get("earnings_per_share", 0)

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if (free_cash_flow_per_share and earnings_per_share and
                free_cash_flow_per_share > earnings_per_share * 0.8):  # Strong FCF conversion
            health_score += 1

        signals.append('bullish' if health_score >=
                                    2 else 'bearish' if health_score == 0 else 'neutral')
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (
                           f"Current Ratio: {metrics.get('current_ratio', 0):.2f}" if metrics.get(
                               "current_ratio") is not None else "Current Ratio: N/A"
                       ) + ", " + (
                           f"D/E: {metrics.get('debt_to_equity', 0):.2f}" if metrics.get(
                               "debt_to_equity") is not None else "D/E: N/A"
                       )
        }

        # 4. Price to X ratios
        pe_ratio = metrics.get("pe_ratio", 0)
        price_to_book = metrics.get("price_to_book", 0)
        price_to_sales = metrics.get("price_to_sales", 0)

        thresholds = [
            (pe_ratio, 25),  # Reasonable P/E ratio
            (price_to_book, 3),  # Reasonable P/B ratio
            (price_to_sales, 5)  # Reasonable P/S ratio
        ]
        price_ratio_score = sum(
            metric is not None and metric < threshold
            for metric, threshold in thresholds
        )

        signals.append('bullish' if price_ratio_score >=
                                    2 else 'bearish' if price_ratio_score == 0 else 'neutral')
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (
                           f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A"
                       ) + ", " + (
                           f"P/B: {price_to_book:.2f}" if price_to_book else "P/B: N/A"
                       ) + ", " + (
                           f"P/S: {price_to_sales:.2f}" if price_to_sales else "P/S: N/A"
                       )
        }

        # Determine overall signal
        bullish_signals = signals.count('bullish')
        bearish_signals = signals.count('bearish')

        if bullish_signals > bearish_signals:
            overall_signal = 'bullish'
        elif bearish_signals > bullish_signals:
            overall_signal = 'bearish'
        else:
            overall_signal = 'neutral'

        # Calculate confidence level
        total_signals = len(signals)
        confidence = max(bullish_signals, bearish_signals) / total_signals

        message_content = {
            "signal": overall_signal,
            "confidence": f"{round(confidence * 100)}%",
            "reasoning": reasoning
        }

        return message_content

    @staticmethod
    def snetiment_analysis(data):
        pass


    @staticmethod
    def researcher(data, perspective: str = "bullish"):
        """
        分析师代理，根据指定视角分析市场数据并生成投资论点

        Args:
            state: 代理状态对象
            perspective: 分析视角，"bullish"(看多) 或 "bearish"(看空)

        Returns:
            包含分析结果的字典
        """
        is_bullish = perspective == "bullish"
        researcher_name = "多方研究员" if is_bullish else "空方研究员"


        # 获取各分析师消息
        # technical_message = next(
        #     msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
        # fundamentals_message = next(
        #     msg for msg in state["messages"] if msg.name == "fundamentals_agent")
        # sentiment_message = next(
        #     msg for msg in state["messages"] if msg.name == "sentiment_agent")
        # valuation_message = next(
        #     msg for msg in state["messages"] if msg.name == "valuation_agent")

        # 解析消息内容

        fundamental_signals = json.loads(data.fundamentals_message.content)
        technical_signals = json.loads(data.technical_message.content)
        sentiment_signals = json.loads(data.sentiment_message.content)
        valuation_signals = json.loads(data.valuation_message.content)


        # 根据视角分析各维度
        thesis_points = []
        confidence_scores = []

        # 技术分析
        tech_signal_match = technical_signals["signal"] == ("bullish" if is_bullish else "bearish")
        if tech_signal_match:
            thesis_points.append(
                f"Technical indicators show {'bullish' if is_bullish else 'bearish'} momentum with {technical_signals['confidence']} confidence")
            confidence_scores.append(
                float(str(technical_signals["confidence"]).replace("%", "")) / 100)
        else:
            if is_bullish:
                thesis_points.append(
                    "Technical indicators may be conservative, presenting buying opportunities")
            else:
                thesis_points.append(
                    "Technical rally may be temporary, suggesting potential reversal")
            confidence_scores.append(0.3)

        # 基本面分析
        fund_signal_match = fundamental_signals["signal"] == ("bullish" if is_bullish else "bearish")
        if fund_signal_match:
            if is_bullish:
                thesis_points.append(
                    f"Strong fundamentals with {fundamental_signals['confidence']} confidence")
            else:
                thesis_points.append(
                    f"Concerning fundamentals with {fundamental_signals['confidence']} confidence")
            confidence_scores.append(
                float(str(fundamental_signals["confidence"]).replace("%", "")) / 100)
        else:
            if is_bullish:
                thesis_points.append(
                    "Company fundamentals show potential for improvement")
            else:
                thesis_points.append(
                    "Current fundamental strength may not be sustainable")
            confidence_scores.append(0.3)

        # 情绪分析
        sent_signal_match = sentiment_signals["signal"] == ("bullish" if is_bullish else "bearish")
        if sent_signal_match:
            if is_bullish:
                thesis_points.append(
                    f"Positive market sentiment with {sentiment_signals['confidence']} confidence")
            else:
                thesis_points.append(
                    f"Negative market sentiment with {sentiment_signals['confidence']} confidence")
            confidence_scores.append(
                float(str(sentiment_signals["confidence"]).replace("%", "")) / 100)
        else:
            if is_bullish:
                thesis_points.append(
                    "Market sentiment may be overly pessimistic, creating value opportunities")
            else:
                thesis_points.append(
                    "Market sentiment may be overly optimistic, indicating potential risks")
            confidence_scores.append(0.3)

        # 估值分析
        val_signal_match = valuation_signals["signal"] == ("bullish" if is_bullish else "bearish")
        if val_signal_match:
            if is_bullish:
                thesis_points.append(
                    f"Stock appears undervalued with {valuation_signals['confidence']} confidence")
            else:
                thesis_points.append(
                    f"Stock appears overvalued with {valuation_signals['confidence']} confidence")
            confidence_scores.append(
                float(str(valuation_signals["confidence"]).replace("%", "")) / 100)
        else:
            if is_bullish:
                thesis_points.append(
                    "Current valuation may not fully reflect growth potential")
            else:
                thesis_points.append(
                    "Current valuation may not fully reflect downside risks")
            confidence_scores.append(0.3)

        # 计算平均置信度
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # 构建消息内容
        message_content = {
            "perspective": perspective,
            "confidence": avg_confidence,
            "thesis_points": thesis_points,
            "reasoning": f"{'Bullish' if is_bullish else 'Bearish'} thesis based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors"
        }


        return message_content


    def researcher_bull(self, data):
        """多方研究员代理"""
        return DataProcess.researcher(data, "bullish")


    def researcher_bear(self, data):
        """空方研究员代理"""
        return DataProcess.researcher(data, "bearish")


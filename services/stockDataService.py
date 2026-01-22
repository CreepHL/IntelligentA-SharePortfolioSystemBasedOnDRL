import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import akshare as ak
import requests

from core.config import STOCK_FILTER_CONFIG
from services.asyncDataService import batch_get_stock_data_sync
from services.lstmProcess import async_lstm_all_stock
from tools.logConfig import log_config

logger = log_config('api')


class StockDataService:

    # def stock_data(self, symbol, period='10y'):
    #     stock = yf.Ticker(symbol)
    #     stock.info
    #     data = stock.history(period=period)
    #     if symbol.startswith('^'):
    #         info = stock.info
    #         period = '1h'
    #     else:
    #         info = stock.info
    #         period = period
    #
    #     print(f"Company Name: {info.get('longName', 'N/A')}")
    #     print(f"Stock Price: {info.get('currentPrice', 'N/A')}")
    #     print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    #     print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    #     print(f"Sector: {info.get('sector', 'N/A')}")
    #     print(f"Industry: {info.get('industry', 'N/A')}")
    #     return data, info
    #
    # def fetch_live_data(self, symbol, api_key):
    #     api_key = "771O3VPDZ5UH78E3"
    #     """Fetches live stock data using Alpha Vantage API."""
    #     try:
    #         url = 'https://www.alphavantage.co/query'
    #         params = {
    #             "function": "TIME_SERIES_DAILY",
    #             "symbol": symbol,
    #             "apikey": api_key,
    #             "outputsize": "compact"  #
    #         }
    #         response = requests.get(url, params=params)
    #         response.raise_for_status()
    #         data = response.json()
    #
    #         if "Time Series (Daily)" in data:
    #             # Convert the data to a pandas DataFrame
    #             df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    #             df.index = pd.to_datetime(df.index)
    #             df.sort_index(inplace=True)  # Sort by date
    #             df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    #             df = df.astype(float)  # Convert columns to numeric
    #
    #             return df
    #         else:
    #             print(
    #                 f"Error fetching live data for {symbol}: {data.get('Error Message', 'No error message provided')}")
    #             return pd.DataFrame()
    #     except requests.exceptions.RequestException as e:
    #         print(f"Request failed for {symbol}: {e}")
    #         return pd.DataFrame()
    #     except (ValueError, KeyError) as e:
    #         print(f"Error parsing live data for {symbol}: {e}")
    #         return pd.DataFrame()

    def financial_metrics(self, symbol):
        """获取财务指标数据"""
        logger.info(f"Getting financial indicators for {symbol}...")
        try:
            # 获取实时行情数据（用于市值和估值比率）
            logger.info("Fetching real-time quotes...")
            realtime_data = ak.stock_zh_a_spot_em()
            if realtime_data is None or realtime_data.empty:
                logger.warning("No real-time quotes data available")
                return [{}]

            stock_data = realtime_data[realtime_data['代码'] == symbol]
            if stock_data.empty:
                logger.warning(f"No real-time quotes found for {symbol}")
                return [{}]

            stock_data = stock_data.iloc[0]
            logger.info("✓ Real-time quotes fetched")

            # 获取新浪财务指标
            logger.info("Fetching Sina financial indicators...")
            current_year = datetime.now().year
            financial_data = ak.stock_financial_analysis_indicator(
                symbol=symbol, start_year=str(current_year - 1))
            if financial_data is None or financial_data.empty:
                logger.warning("No financial indicator data available")
                return [{}]

            # 按日期排序并获取最新的数据
            financial_data['日期'] = pd.to_datetime(financial_data['日期'])
            financial_data = financial_data.sort_values('日期', ascending=False)
            latest_financial = financial_data.iloc[0] if not financial_data.empty else pd.Series(
            )
            logger.info(
                f"✓ Financial indicators fetched ({len(financial_data)} records)")
            logger.info(f"Latest data date: {latest_financial.get('日期')}")

            # 获取利润表数据（用于计算 price_to_sales）
            logger.info("Fetching income statement...")
            try:
                income_statement = ak.stock_financial_report_sina(
                    stock=f"sh{symbol}", symbol="利润表")
                if not income_statement.empty:
                    latest_income = income_statement.iloc[0]
                    logger.info("✓ Income statement fetched")
                else:
                    logger.warning("Failed to get income statement")
                    logger.error("No income statement data found")
                    latest_income = pd.Series()
            except Exception as e:
                logger.warning("Failed to get income statement")
                logger.error(f"Error getting income statement: {e}")
                latest_income = pd.Series()

            # 构建完整指标数据
            logger.info("Building indicators...")
            try:
                def convert_percentage(value: float) -> float:
                    """将百分比值转换为小数"""
                    try:
                        return float(value) / 100.0 if value is not None else 0.0
                    except:
                        return 0.0

                all_metrics = {
                    # 市场数据
                    "market_cap": float(stock_data.get("总市值", 0)),
                    "float_market_cap": float(stock_data.get("流通市值", 0)),

                    # 盈利数据
                    "revenue": float(latest_income.get("营业总收入", 0)),
                    "net_income": float(latest_income.get("净利润", 0)),
                    "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
                    "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
                    "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),

                    # 增长指标
                    "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
                    "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
                    "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),

                    # 财务健康指标
                    "current_ratio": float(latest_financial.get("流动比率", 0)),
                    "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
                    "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)),
                    "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)),

                    # 估值比率
                    "pe_ratio": float(stock_data.get("市盈率-动态", 0)),
                    "price_to_book": float(stock_data.get("市净率", 0)),
                    "price_to_sales": float(stock_data.get("总市值", 0)) / float(
                        latest_income.get("营业总收入", 1)) if float(latest_income.get("营业总收入", 0)) > 0 else 0,
                }

                # 只返回 agent 需要的指标
                agent_metrics = {
                    # 盈利能力指标
                    "return_on_equity": all_metrics["return_on_equity"],
                    "net_margin": all_metrics["net_margin"],
                    "operating_margin": all_metrics["operating_margin"],

                    # 增长指标
                    "revenue_growth": all_metrics["revenue_growth"],
                    "earnings_growth": all_metrics["earnings_growth"],
                    "book_value_growth": all_metrics["book_value_growth"],

                    # 财务健康指标
                    "current_ratio": all_metrics["current_ratio"],
                    "debt_to_equity": all_metrics["debt_to_equity"],
                    "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
                    "earnings_per_share": all_metrics["earnings_per_share"],

                    # 估值比率
                    "pe_ratio": all_metrics["pe_ratio"],
                    "price_to_book": all_metrics["price_to_book"],
                    "price_to_sales": all_metrics["price_to_sales"],
                }

                logger.info("✓ Indicators built successfully")

                # 打印所有获取到的指标数据（用于调试）
                logger.debug("\n获取到的完整指标数据：")
                for key, value in all_metrics.items():
                    logger.debug(f"{key}: {value}")

                logger.debug("\n传递给 agent 的指标数据：")
                for key, value in agent_metrics.items():
                    logger.debug(f"{key}: {value}")

                return [agent_metrics]

            except Exception as e:
                logger.error(f"Error building indicators: {e}")
                return [{}]

        except Exception as e:
            logger.error(f"Error getting financial indicators: {e}")
            return [{}]

    def financial_statements(self, symbol: str) -> Dict[str, Any]:
        """获取财务报表数据"""
        logger.info(f"Getting financial statements for {symbol}...")
        try:
            # 获取资产负债表数据
            logger.info("Fetching balance sheet...")
            try:
                balance_sheet = ak.stock_financial_report_sina(
                    stock=f"sh{symbol}", symbol="资产负债表")
                if not balance_sheet.empty:
                    latest_balance = balance_sheet.iloc[0]
                    previous_balance = balance_sheet.iloc[1] if len(
                        balance_sheet) > 1 else balance_sheet.iloc[0]
                    logger.info("✓ Balance sheet fetched")
                else:
                    logger.warning("Failed to get balance sheet")
                    logger.error("No balance sheet data found")
                    latest_balance = pd.Series()
                    previous_balance = pd.Series()
            except Exception as e:
                logger.warning("Failed to get balance sheet")
                logger.error(f"Error getting balance sheet: {e}")
                latest_balance = pd.Series()
                previous_balance = pd.Series()

            # 获取利润表数据
            logger.info("Fetching income statement...")
            try:
                income_statement = ak.stock_financial_report_sina(
                    stock=f"sh{symbol}", symbol="利润表")
                if not income_statement.empty:
                    latest_income = income_statement.iloc[0]
                    previous_income = income_statement.iloc[1] if len(
                        income_statement) > 1 else income_statement.iloc[0]
                    logger.info("✓ Income statement fetched")
                else:
                    logger.warning("Failed to get income statement")
                    logger.error("No income statement data found")
                    latest_income = pd.Series()
                    previous_income = pd.Series()
            except Exception as e:
                logger.warning("Failed to get income statement")
                logger.error(f"Error getting income statement: {e}")
                latest_income = pd.Series()
                previous_income = pd.Series()

            # 获取现金流量表数据
            logger.info("Fetching cash flow statement...")
            try:
                cash_flow = ak.stock_financial_report_sina(
                    stock=f"sh{symbol}", symbol="现金流量表")
                if not cash_flow.empty:
                    latest_cash_flow = cash_flow.iloc[0]
                    previous_cash_flow = cash_flow.iloc[1] if len(
                        cash_flow) > 1 else cash_flow.iloc[0]
                    logger.info("✓ Cash flow statement fetched")
                else:
                    logger.warning("Failed to get cash flow statement")
                    logger.error("No cash flow data found")
                    latest_cash_flow = pd.Series()
                    previous_cash_flow = pd.Series()
            except Exception as e:
                logger.warning("Failed to get cash flow statement")
                logger.error(f"Error getting cash flow statement: {e}")
                latest_cash_flow = pd.Series()
                previous_cash_flow = pd.Series()

            # 构建财务数据
            line_items = []
            try:
                # 处理最新期间数据
                current_item = {
                    # 从利润表获取
                    "net_income": float(latest_income.get("净利润", 0)),
                    "operating_revenue": float(latest_income.get("营业总收入", 0)),
                    "operating_profit": float(latest_income.get("营业利润", 0)),

                    # 从资产负债表计算营运资金
                    "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(
                        latest_balance.get("流动负债合计", 0)),

                    # 从现金流量表获取
                    "depreciation_and_amortization": float(
                        latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                    "capital_expenditure": abs(
                        float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                    "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(
                        float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
                }
                line_items.append(current_item)
                logger.info("✓ Latest period data processed successfully")

                # 处理上一期间数据
                previous_item = {
                    "net_income": float(previous_income.get("净利润", 0)),
                    "operating_revenue": float(previous_income.get("营业总收入", 0)),
                    "operating_profit": float(previous_income.get("营业利润", 0)),
                    "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(
                        previous_balance.get("流动负债合计", 0)),
                    "depreciation_and_amortization": float(
                        previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                    "capital_expenditure": abs(
                        float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                    "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(
                        float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
                }
                line_items.append(previous_item)
                logger.info("✓ Previous period data processed successfully")

            except Exception as e:
                logger.error(f"Error processing financial data: {e}")
                default_item = {
                    "net_income": 0,
                    "operating_revenue": 0,
                    "operating_profit": 0,
                    "working_capital": 0,
                    "depreciation_and_amortization": 0,
                    "capital_expenditure": 0,
                    "free_cash_flow": 0
                }
                line_items = [default_item, default_item]

            return line_items

        except Exception as e:
            logger.error(f"Error getting financial statements: {e}")
            default_item = {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
            return [default_item, default_item]


    def macro_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        try:
            # 获取实时行情
            realtime_data = ak.stock_zh_a_spot_em()
            stock_data = realtime_data[realtime_data['代码'] == symbol].iloc[0]

            return {
                "market_cap": float(stock_data.get("总市值", 0)),
                "volume": float(stock_data.get("成交量", 0)),
                # A股没有平均成交量，暂用当日成交量
                "average_volume": float(stock_data.get("成交量", 0)),
                "fifty_two_week_high": float(stock_data.get("52周最高", 0)),
                "fifty_two_week_low": float(stock_data.get("52周最低", 0))
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    def calculate_strength_score(self, stock_data: Dict) -> Dict:
        """计算股票强势分数 - 增强版,包含基本面评分"""
        score_breakdown = {
            'technical': 0,    # 技术面 (30分)
            'valuation': 0,    # 估值 (25分)
            'profitability': 0,  # 盈利质量 (30分)
            'safety': 0,       # 安全性 (10分)
            'dividend': 0      # 分红 (5分)
        }
        try:
            # ===== 1. 技术面得分 (30分) =====
            # 1.1 涨跌幅 (10分)
            change_pct = stock_data.get('change_pct', 0)
            if change_pct > 5:
                score_breakdown['technical'] += 10
            elif change_pct > 2:
                score_breakdown['technical'] += 7
            elif change_pct > 0:
                score_breakdown['technical'] += 4
            elif change_pct > -2:
                score_breakdown['technical'] += 2
            # 1.2 动量 (15分)
            momentum = stock_data.get('momentum_20d', 0)
            if momentum > 15:
                score_breakdown['technical'] += 15
            elif momentum > 10:
                score_breakdown['technical'] += 12
            elif momentum > 5:
                score_breakdown['technical'] += 8
            elif momentum > 0:
                score_breakdown['technical'] += 4
            # 1.3 流动性 (5分) - 基于换手率
            # 使用换手率而非成交额，更公平地评估流动性（消除市值影响）
            turnover_rate = stock_data.get('turnover_rate', 0)
            if turnover_rate and 1 <= turnover_rate < 3:  # 最佳流动性：适中活跃
                score_breakdown['technical'] += 5
            elif turnover_rate and 3 <= turnover_rate < 5:  # 良好流动性
                score_breakdown['technical'] += 4
            elif turnover_rate and 5 <= turnover_rate < 8:  # 流动性充足但略偏高
                score_breakdown['technical'] += 3
            elif turnover_rate and 0.5 <= turnover_rate < 1:  # 流动性偏低但可接受
                score_breakdown['technical'] += 2
            elif turnover_rate and turnover_rate >= 8:  # 换手过高，投机性强
                score_breakdown['technical'] += 1
            # 换手率 < 0.5% 流动性不足，不得分
            # ===== 2. 估值得分 (25分) =====
            # 2.1 PE估值 (10分) - 按10,20,30区分
            pe = stock_data.get('pe_ratio', 0)
            if pe and 0 < pe < 10:
                score_breakdown['valuation'] += 10
            elif pe and 10 <= pe < 20:
                score_breakdown['valuation'] += 7
            elif pe and 20 <= pe < 30:
                score_breakdown['valuation'] += 4
            # PE >= 30 不得分
            # 2.2 PB估值 (10分) - 从"便宜"角度评分,范围宽松
            pb = stock_data.get('pb_ratio', 0)
            if pb and 0 < pb < 2:      # 低估
                score_breakdown['valuation'] += 10
            elif pb and 2 <= pb < 4:   # 合理
                score_breakdown['valuation'] += 8
            elif pb and 4 <= pb < 7:   # 适中
                score_breakdown['valuation'] += 5
            elif pb and 7 <= pb < 10:  # 偏高
                score_breakdown['valuation'] += 2
            # PB >= 10 不得分
            # 2.3 PEG (5分) - 降低权重,因为是估算值
            peg = stock_data.get('peg', 0)
            if peg and 0 < peg < 1:
                score_breakdown['valuation'] += 5
            elif peg and 1 <= peg < 1.5:
                score_breakdown['valuation'] += 3
            elif peg and 1.5 <= peg < 2:
                score_breakdown['valuation'] += 1
            # PEG >= 2 不得分
            # ===== 3. 盈利质量得分 (30分) - 核心 =====
            # 3.1 ROE (15分)
            roe = stock_data.get('roe', 0)
            if roe and roe > 20:
                score_breakdown['profitability'] += 15
            elif roe and roe > 15:
                score_breakdown['profitability'] += 12
            elif roe and roe > 10:
                score_breakdown['profitability'] += 8
            elif roe and roe > 5:
                score_breakdown['profitability'] += 4
            # 3.2 净利润增长率 (15分)
            profit_growth = stock_data.get('profit_growth', 0)
            if profit_growth and profit_growth > 30:
                score_breakdown['profitability'] += 15
            elif profit_growth and profit_growth > 20:
                score_breakdown['profitability'] += 12
            elif profit_growth and profit_growth > 10:
                score_breakdown['profitability'] += 8
            elif profit_growth and profit_growth > 0:
                score_breakdown['profitability'] += 4
            # ===== 4. 安全性得分 (10分) - 基于现有数据的简化评分 =====
            # 由于资产负债率等财务数据无法获取,使用现有指标构建安全性评分
            # 4.1 基于PB的安全边际 (3分) - 从"安全"角度评分,范围严格
            pb = stock_data.get('pb_ratio', 0)
            if pb and 0 < pb < 1.0:  # 破净,极度安全
                score_breakdown['safety'] += 3
            elif pb and 1.0 <= pb < 1.5:  # 接近破净,很安全
                score_breakdown['safety'] += 2
            elif pb and 1.5 <= pb < 2.5:  # 低估值区,有安全边际
                score_breakdown['safety'] += 1
            # PB >= 2.5 安全性不加分
            # 4.2 基于股息率的稳定性 (3分)
            div_yield = stock_data.get('dividend_yield', 0)
            if div_yield and div_yield > 5:  # 高分红,经营稳定
                score_breakdown['safety'] += 3
            elif div_yield and div_yield > 3:
                score_breakdown['safety'] += 2
            elif div_yield and div_yield > 1:
                score_breakdown['safety'] += 1
            # 股息率 <= 1% 不得分
            # 4.3 基于换手率的波动性 (4分) - 增加权重
            turnover_rate = stock_data.get('turnover_rate', 0)
            if turnover_rate and 0 < turnover_rate < 2:  # 低换手,筹码稳定
                score_breakdown['safety'] += 4
            elif turnover_rate and 2 <= turnover_rate < 5:  # 换手适中
                score_breakdown['safety'] += 3
            elif turnover_rate and 5 <= turnover_rate < 10:  # 换手偏高
                score_breakdown['safety'] += 1
            # 换手率 >= 10% 说明投机性强,不得分
            # ===== 5. 分红得分 (5分) =====
            dividend_yield = stock_data.get('dividend_yield', 0)
            if dividend_yield and dividend_yield > 5:
                score_breakdown['dividend'] += 5
            elif dividend_yield and dividend_yield > 3:
                score_breakdown['dividend'] += 4
            elif dividend_yield and dividend_yield > 2:
                score_breakdown['dividend'] += 3
            elif dividend_yield and dividend_yield > 1:
                score_breakdown['dividend'] += 2
            elif dividend_yield and dividend_yield > 0.5:
                score_breakdown['dividend'] += 1
            # 股息率 <= 0.5% 不得分

        except Exception as e:
            logger.error(f"计算强势分数失败: {e}")
            return {'total': 0, 'breakdown': score_breakdown, 'grade': 'D'}

        total_score = sum(score_breakdown.values())
        grade = self._get_grade(total_score)

        return {
            'total': total_score,
            'breakdown': score_breakdown,
            'grade': grade
        }

    def _get_grade(self, score: float) -> str:

        """根据分数获取评级"""

        if score >= 85:

            return 'A+'

        elif score >= 75:

            return 'A'

        elif score >= 65:

            return 'B+'

        elif score >= 55:

            return 'B'

        elif score >= 45:

            return 'C'

        else:

            return 'D'

    def filter_by_pe_ratio(self, stocks_data: List[Dict]) -> List[Dict]:
        """按市盈率筛选股票"""
        filtered_stocks = []

        for stock in stocks_data:
            try:
                pe_ratio = stock.get('pe_ratio', 0)
                # 修复PE筛选问题：确保PE值是数字类型，如果是None则设为0
                if pe_ratio is None:
                    pe_ratio = 0

                # 过滤掉PE为0或负数的股票（可能是亏损股）
                if 0 < pe_ratio <= STOCK_FILTER_CONFIG['max_pe_ratio']:
                    filtered_stocks.append(stock)
            except Exception as e:
                logger.error(f"PE筛选失败 {stock.get('code', 'unknown')}: {e}")
                continue

        logger.info(f"PE筛选后剩余 {len(filtered_stocks)} 只股票")
        return filtered_stocks

    def filter_by_strength(self, stocks_data: List[Dict]) -> List[Dict]:
        """按强势指标筛选股票"""
        try:
            # 计算每只股票的强势分数
            for stock in stocks_data:
                score_result = self.calculate_strength_score(stock)
                stock['strength_score_detail'] = score_result  # 保存详细评分
                stock['strength_score'] = score_result['total']  # 保存总分
                stock['strength_grade'] = score_result['grade']  # 保存评级

            # 按强势分数排序，选择前N只
            sorted_stocks = sorted(stocks_data,
                                   key=lambda x: x['strength_score'],
                                   reverse=True)

            # 根据配置的最小强势分数筛选
            min_score = STOCK_FILTER_CONFIG.get('min_strength_score', 45)  # 降低到45分
            strong_stocks = [stock for stock in sorted_stocks if stock['strength_score'] >= min_score]

            logger.info(f"强势筛选后剩余 {len(strong_stocks)} 只股票")
            return strong_stocks

        except Exception as e:
            logger.error(f"强势筛选失败: {e}")
            return []

    def apply_additional_filters(self, stocks_data: List[Dict]) -> List[Dict]:
        """应用额外的筛选条件"""
        filtered_stocks = []

        for stock in stocks_data:
            try:
                # 过滤条件
                price = stock.get('price', 0)
                turnover_rate = stock.get('turnover_rate', 0)
                change_pct = stock.get('change_pct', 0)

                # 排除停牌股票（涨跌幅为0且换手率很小）
                if change_pct == 0 and (turnover_rate is None or turnover_rate < 0.1):  # 0.1%换手率
                    continue

                # 排除价格过低的股票
                if price < STOCK_FILTER_CONFIG['min_price']:
                    continue

                # 排除换手率过小的股票
                min_turnover_rate = STOCK_FILTER_CONFIG.get('min_turnover_rate', 0.5)  # 默认0.5%
                if turnover_rate is None or turnover_rate < min_turnover_rate:
                    continue

                # 排除跌停股票
                if change_pct <= -9.8:
                    continue

                filtered_stocks.append(stock)

            except Exception as e:
                logger.error(f"附加筛选失败 {stock.get('code', 'unknown')}: {e}")
                continue

        logger.info(f"附加筛选后剩余 {len(filtered_stocks)} 只股票")
        return filtered_stocks


    def _generate_selection_reason(self, stock: Dict) -> str:

        """生成选择理由 - 包含基本面指标"""
        reasons = []
        # 基本信息
        pe_ratio = stock.get('pe_ratio', 0)
        pb_ratio = stock.get('pb_ratio', 0)
        change_pct = stock.get('change_pct', 0)
        momentum = stock.get('momentum_20d', 0)
        strength_score = stock.get('strength_score', 0)
        strength_grade = stock.get('strength_grade', '')
        # 基本面指标
        roe = stock.get('roe', 0)
        profit_growth = stock.get('profit_growth', 0)
        dividend_yield = stock.get('dividend_yield', 0)
        turnover_rate = stock.get('turnover_rate', 0)
        # 估值
        if pe_ratio:
            reasons.append(f"PE={pe_ratio:.2f}")
        if pb_ratio:
            reasons.append(f"PB={pb_ratio:.2f}")
        # 技术面
        if change_pct > 3:
            reasons.append("当日强势上涨")
        elif change_pct > 0:
            reasons.append("当日上涨")
        if momentum > 10:
            reasons.append("20日动量强劲")
        elif momentum > 0:
            reasons.append("20日动量向上")
        # 基本面
        if roe and roe > 15:
            reasons.append(f"ROE优秀({roe:.1f}%)")
        elif roe and roe > 10:
            reasons.append(f"ROE良好({roe:.1f}%)")
        if profit_growth and profit_growth > 20:
            reasons.append(f"高成长({profit_growth:.1f}%)")
        elif profit_growth and profit_growth > 10:
            reasons.append(f"成长性好({profit_growth:.1f}%)")
        # 安全性 - 基于新的评分逻辑
        safety_score = stock.get('strength_score_detail', {}).get('breakdown', {}).get('safety', 0)
        if safety_score >= 8:
            reasons.append("安全性高")
        elif safety_score >= 6:
            reasons.append("安全性良好")
        # 综合评分
        reasons.append(f"综合{strength_grade}级({strength_score:.1f}分)")
        return "；".join(reasons)

    def select_top_stocks(self, stocks_data: List[Dict]) -> List[Dict]:
        """选择最终的推荐股票 - 直接按分数排序选择，不限制行业"""
        try:
            # 0. 去重（防止输入数据中有重复）
            unique_stocks = {}
            for stock in stocks_data:
                code = stock.get('code')
                if code and code not in unique_stocks:
                    unique_stocks[code] = stock

            stocks_data = list(unique_stocks.values())
            logger.info(f"去重后股票数量: {len(stocks_data)}")

            # 1. 首先按PE筛选
            pe_filtered = self.filter_by_pe_ratio(stocks_data)

            # 2. 应用额外筛选条件
            additional_filtered = self.apply_additional_filters(pe_filtered)

            # 3. 按强势筛选并排序
            strength_filtered = self.filter_by_strength(additional_filtered)

            # 4. 直接按分数排序，不限制行业
            strength_filtered.sort(key=lambda x: x['strength_score'], reverse=True)

            # 5. 选择前N只股票
            final_selection = strength_filtered[:STOCK_FILTER_CONFIG['max_stocks']]

            # 6. 添加选择理由和排名
            for i, stock in enumerate(final_selection):
                stock['rank'] = i + 1
                stock['selection_reason'] = self._generate_selection_reason(stock)

            logger.info(f"最终选择 {len(final_selection)} 只股票 (不限制行业)")
            return final_selection

        except Exception as e:
            logger.error(f"股票选择失败: {e}")
            return []

    def csi300_data(self):
        """加载沪深300成分股列表 - 优先使用本地缓存"""
        try:
            # 方法1: 从本地JSON文件加载(快速)
            local_file = './data/csi300_stocks.json'
            if os.path.exists(local_file):
                logger.info("从本地文件加载沪深300成分股列表...")
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stocks = data['stocks']
                    logger.info(f"成功从本地加载 {len(stocks)} 只沪深300成分股 (更新日期: {data.get('update_date', '未知')})")
                    return pd.DataFrame(stocks)

            # 方法2: 使用akshare在线获取(慢速,作为备用)
            logger.warning("本地文件不存在,尝试在线获取沪深300成分股列表(可能较慢)...")
            csi300_stocks = ak.index_stock_cons(symbol="000300")
            if not csi300_stocks.empty:
                logger.info(f"在线获取成功: {len(csi300_stocks)} 只")
                # 保存到本地以便下次使用
                result = pd.DataFrame({
                    'code': csi300_stocks['品种代码'].tolist(),
                    'name': csi300_stocks['品种名称'].tolist()
                })
                # 保存到本地
                os.makedirs('./data', exist_ok=True)
                save_data = {
                    'update_date': datetime.now().strftime('%Y-%m-%d'),
                    'note': '沪深300成分股列表 - 自动生成',
                    'stocks': result.to_dict('records')
                }
                with open(local_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存到本地文件: {local_file}")
                return result

            logger.error("无法获取沪深300成分股列表")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"加载沪深300成分股列表失败: {e}")
            return pd.DataFrame()

    def market_all_data(self, symbols, data):
        # 获取沪深300成分股列表（优先使用本地缓存）
        a_share_list = self.csi300_data()
        stock_codes = a_share_list['code'].tolist()
        # performance utility
        all_stock_data = batch_get_stock_data_sync(
                    stock_codes,
                    calculate_momentum=True,
                    include_fundamental=True,
                    max_concurrent=20  # 可以调整并发数
                )
        # 初步筛选出的股票列表
        selected_stocks = self.select_top_stocks(all_stock_data)
        code_list = [stock['code'] for stock in selected_stocks if 'code' in stock]
        # LSTM收益率预测
        asyncio.run(async_lstm_all_stock(code_list))
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        end_date = data.get("end_date", yesterday.strftime('%Y-%m-%d'))

        # Ensure end_date is not in the future
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        if end_date_obj > yesterday:
            end_date = yesterday.strftime('%Y-%m-%d')
            end_date_obj = yesterday

        start_date = data.get("start_date", (end_date_obj - timedelta(days=365)).strftime('%Y-%m-%d'))

        # stock price data
        prices_df = self.history_price(symbols, start_date, end_date)
        financial_metrics = self.financial_metrics(symbols)
        financial_line_items = self.financial_statements(symbols)
        macro_data = self.macro_data(symbols)
        prices_dict = prices_df.to_dict('records')
        # 保存推理信息到metadata供API使用
        market_data = {
            "data": {
                "ticker": symbols,
                "prices": prices_dict,
                "start_date": start_date,
                "end_date": end_date,
                "financial_metrics": financial_metrics,
                "financial_line_items": financial_line_items,
                "market_cap": macro_data.get("market_cap", 0),
                "market_data": macro_data,
            },
        }
        return market_data


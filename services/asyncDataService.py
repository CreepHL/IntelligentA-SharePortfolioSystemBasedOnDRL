import asyncio
import random
from typing import List, Dict, Optional

import aiohttp

from tools.logConfig import log_config

logger = log_config('api')


class AsyncDataService:

    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = None  # 将在异步上下文中初始化
        self.failed_stocks = []
        self._hist_cache = {}  # 历史数据缓存

        # User-Agent池
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        ]

    def _get_random_user_agent(self) -> str:
        """随机获取User-Agent"""
        return random.choice(self.user_agents)

    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str,
                                max_retries: int = 3, timeout: int = 10) -> Optional[str]:
        """
        带重试的异步HTTP请求

        Args:
            session: aiohttp会话
            url: 请求URL
            max_retries: 最大重试次数
            timeout: 超时时间(秒)

        Returns:
            响应文本或None
        """
        headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://gu.qq.com/'
        }

        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()

                    # 如果状态码不是200,等待后重试
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (2 ** attempt))

            except asyncio.TimeoutError:
                logger.debug(f"请求超时 (尝试 {attempt + 1}/{max_retries}): {url[:80]}...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                logger.debug(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        return None

    async def get_realtime_data(self, session: aiohttp.ClientSession,
                                stock_code: str) -> Dict:
        """异步获取股票实时数据"""
        # 使用信号量控制并发
        async with self.semaphore:
            try:
                # 构造腾讯财经API请求
                if stock_code.startswith('6'):
                    symbol = f"sh{stock_code}"
                else:
                    symbol = f"sz{stock_code}"

                url = f"https://qt.gtimg.cn/q={symbol}"

                content = await self._fetch_with_retry(session, url, max_retries=3, timeout=10)

                if content and 'v_' in content:
                    data_str = content.split('"')[1]
                    data_parts = data_str.split('~')

                    if len(data_parts) > 35:
                        name = data_parts[1]
                        price = float(data_parts[3]) if data_parts[3] and data_parts[3] != '' else 0
                        prev_close = float(data_parts[4]) if data_parts[4] and data_parts[4] != '' else 0
                        change_pct = float(data_parts[32]) if data_parts[32] and data_parts[32] != '' else 0
                        volume = int(float(data_parts[6])) if data_parts[6] and data_parts[6] != '' else 0
                        turnover = int(float(data_parts[7])) if data_parts[7] and data_parts[7] != '' else 0

                        # 获取总市值和总股本
                        market_cap = None
                        total_shares = None

                        if len(data_parts) > 23 and data_parts[23]:
                            try:
                                market_cap = float(data_parts[23])
                            except ValueError:
                                pass

                        if len(data_parts) > 25 and data_parts[25]:
                            try:
                                total_shares = float(data_parts[25])
                            except ValueError:
                                pass

                        # 获取换手率
                        turnover_rate = None
                        if len(data_parts) > 27 and data_parts[27]:
                            try:
                                turnover_rate = float(data_parts[27])
                            except ValueError:
                                pass

                        # 获取PE值 - 按优先级
                        pe_ratio = None
                        pe_fields = [
                            data_parts[39] if len(data_parts) > 39 else None,  # 基本面PE
                            data_parts[22] if len(data_parts) > 22 else None,  # TTM PE
                            data_parts[15] if len(data_parts) > 15 else None,  # 静态PE
                            data_parts[14] if len(data_parts) > 14 else None  # 动态PE
                        ]

                        for pe_str in pe_fields:
                            if pe_str and pe_str != '':
                                try:
                                    pe_value = float(pe_str)
                                    if 0 < pe_value < 1000:
                                        pe_ratio = pe_value
                                        break
                                except ValueError:
                                    continue

                        # 获取PB值
                        pb_ratio = None
                        if len(data_parts) > 16 and data_parts[16]:
                            try:
                                pb_value = float(data_parts[16])
                                if 0 < pb_value < 100:
                                    pb_ratio = pb_value
                            except ValueError:
                                pass

                        return {
                            'code': stock_code,
                            'name': name,
                            'price': price,
                            'prev_close': prev_close,
                            'change_pct': change_pct,
                            'pe_ratio': pe_ratio,
                            'pb_ratio': pb_ratio,
                            'market_cap': market_cap,
                            'total_shares': total_shares,
                            'volume': volume,
                            'turnover': turnover,
                            'turnover_rate': turnover_rate
                        }
            except Exception as e:
                logger.debug(f"获取股票 {stock_code} 实时数据失败: {e}")

            return {}


    def _calculate_financial_health(self, pb: Optional[float], div_yield: Optional[float],
                                   pe: Optional[float], turnover: Optional[float]) -> int:
        """计算财务健康度评分"""
        score = 50

        try:
            if pb:
                if pb < 1:
                    score += 20
                elif pb < 2:
                    score += 10
                elif pb > 10:
                    score -= 20
                elif pb > 5:
                    score -= 10

            if div_yield:
                if div_yield > 5:
                    score += 15
                elif div_yield > 3:
                    score += 10
                elif div_yield > 2:
                    score += 5
                elif div_yield < 1:
                    score -= 5

            if pe:
                if 10 < pe < 20:
                    score += 10
                elif 20 <= pe < 30:
                    score += 5
                elif pe >= 50:
                    score -= 10

            if turnover:
                if 1 < turnover < 5:
                    score += 5
                elif turnover > 20:
                    score -= 5
        except:
            pass

        return max(0, min(100, score))

    async def get_fundamental_data(self, session: aiohttp.ClientSession,
                                   stock_code: str) -> Dict:
        """异步获取股票基本面数据"""
        async with self.semaphore:
            try:
                # 确定市场代码
                if stock_code.startswith('6') or stock_code.startswith('688'):
                    market = 'sh'
                else:
                    market = 'sz'

                symbol = f"{market}{stock_code}"
                url = f"https://qt.gtimg.cn/q={symbol}"

                content = await self._fetch_with_retry(session, url, max_retries=2, timeout=10)

                if content and 'v_' in content and len(content.split('~')) > 52:
                    data_str = content.split('"')[1]
                    data_parts = data_str.split('~')

                    # 解析PB市净率
                    pb_ratio = None
                    if len(data_parts) > 46 and data_parts[46]:
                        try:
                            pb_ratio = float(data_parts[46])
                            if pb_ratio <= 0:
                                pb_ratio = None
                        except ValueError:
                            pass

                    # 解析股息率
                    dividend_yield = None
                    ######################################
                    manual_dividend = None
                    if manual_dividend is not None:
                        dividend_yield = manual_dividend
                    else:
                        current_price = float(data_parts[3]) if data_parts[3] else None
                        dividend_data = None

                        if len(data_parts) > 53 and data_parts[53]:
                            try:
                                dividend_data = float(data_parts[53])
                                if dividend_data < 0:
                                    dividend_data = None
                            except ValueError:
                                pass

                        if current_price and current_price > 0 and dividend_data and dividend_data > 0:
                            per_share_dividend = dividend_data / 10
                            dividend_yield = (per_share_dividend / current_price) * 100

                            if not (0 < dividend_yield <= 20):
                                dividend_yield = None

                    # 解析换手率
                    turnover_rate = None
                    if len(data_parts) > 56 and data_parts[56]:
                        try:
                            turnover_rate = float(data_parts[56])
                        except ValueError:
                            pass

                    # 获取PE和计算PEG
                    pe_ratio = None
                    peg = None
                    if len(data_parts) > 39 and data_parts[39]:
                        try:
                            pe_value = float(data_parts[39])
                            if 0 < pe_value < 200:
                                pe_ratio = pe_value

                                if pb_ratio:
                                    if pb_ratio < 1:
                                        assumed_growth = 20
                                    elif pb_ratio > 5:
                                        assumed_growth = 10
                                    else:
                                        assumed_growth = 15
                                else:
                                    assumed_growth = 15

                                peg = pe_ratio / assumed_growth
                        except ValueError:
                            pass

                    # 计算ROE
                    roe = None
                    if pb_ratio and pe_ratio and pe_ratio > 0:
                        try:
                            roe = (pb_ratio / pe_ratio) * 100
                            if roe < -50 or roe > 50:
                                roe = None
                        except:
                            roe = None

                    # 估算利润增长率
                    profit_growth = None
                    if roe and dividend_yield:
                        try:
                            payout_ratio = min(dividend_yield / roe, 0.9) if roe > 0 else 0.5
                            profit_growth = roe * (1 - payout_ratio)
                        except:
                            profit_growth = None

                    # 计算财务健康度评分
                    financial_health_score = self._calculate_financial_health(
                        pb_ratio, dividend_yield, pe_ratio, turnover_rate
                    )

                    return {

                        'pb_ratio': pb_ratio,

                        'dividend_yield': dividend_yield,

                        'peg': peg,

                        'turnover_rate': turnover_rate,

                        'financial_health_score': financial_health_score,

                        'roe': roe,

                        'profit_growth': profit_growth,

                        'debt_ratio': None,

                        'current_ratio': None,

                        'gross_margin': None,

                        # 不设置market_cap和total_shares为None，保留实时数据中的值

                    }

            except Exception as e:
                logger.debug(f"获取股票 {stock_code} 基本面数据失败: {e}")

            return {

                'pb_ratio': None,

                'dividend_yield': None,

                'peg': None,

                'turnover_rate': None,

                'financial_health_score': 0,

                'roe': None,

                'profit_growth': None,

                'debt_ratio': None,

                'current_ratio': None,

                'gross_margin': None,

                # 不设置market_cap和total_shares为None，保留实时数据中的值

            }

    async def get_stock_data(self, stock_code: List[str], calculate_momentum: bool = True,
                             include_fundamental: bool = True) -> List[Dict]:
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        logger.info(f"开始批量获取 {len(stock_code)} 只股票数据 (最大并发: {self.max_concurrent})")
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2, limit_per_host=self.max_concurrent,
                                         ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=60, connect=60, sock_read=15)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            realtime_tasks = [self.get_realtime_data(session, code) for code in stock_code]
            realtime_data = await asyncio.gather(*realtime_tasks)
            realtime_data = [data for data in realtime_data if data and data.get('code')]
            if include_fundamental:
                logger.info("开始批量获取股票实时基本面数据")
                fundamental_tasks = [self.get_fundamental_data(session, code) for code in stock_code]
                fundamental_results = await asyncio.gather(*fundamental_tasks)
                for stock, fundamental in zip(realtime_data, fundamental_results):
                    stock.update(fundamental)
        return realtime_data


def batch_get_stock_data_sync(stock_codes: List[str], calculate_momentum: bool = True,
                              include_fundamental: bool = True, max_concurrent: int = 20) -> List[Dict]:
    fetcher = AsyncDataService(max_concurrent=max_concurrent)
    return asyncio.run(
        fetcher.get_stock_data(stock_codes, calculate_momentum, include_fundamental))

"""
数据获取模块 - 交易所API
支持Binance, Bybit, OKX等
"""
import ccxt
import pandas as pd
from datetime import datetime
import os
import numpy as np
import requests


# 默认代理地址
DEFAULT_PROXY = 'http://127.0.0.1:10808'


class BinanceDataFetcher:
    """数据获取器"""

    def __init__(self, symbol="BTC/USDT", timeframe="15m", exchange="binance", proxy=None):
        self.symbol = symbol
        self.timeframe = timeframe

        proxy_url = proxy if proxy else DEFAULT_PROXY

        # 交易所配置
        exchange_configs = {
            'binance': {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            },
            'bybit': {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            },
            'okx': {
                'enableRateLimit': True,
            }
        }

        config = exchange_configs.get(exchange, exchange_configs['binance'])

        # 添加代理到requests session
        if proxy_url:
            session = requests.Session()
            session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            config['session'] = session

        self.exchange = getattr(ccxt, exchange)(config)

    def fetch_ohlcv(self, limit=1000, since=None):
        """
        获取K线数据

        Args:
            limit: 获取的K线数量
            since: 起始时间戳(毫秒)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=limit,
                since=since
            )

            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def fetch_historical_data(self, days=400, limit_per_request=1000):
        """
        获取历史数据（尽可能多）

        Args:
            days: 获取的天数
            limit_per_request: 每次请求的K线数量

        Returns:
            DataFrame with historical OHLCV data
        """
        # 计算起始时间
        since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

        all_data = []
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=limit_per_request,
                    since=since
                )

                if len(ohlcv) == 0:
                    break

                all_data.extend(ohlcv)

                # 更新起始时间
                since = ohlcv[-1][0] + 1

                # 如果获取的数据少于请求的数量，说明数据已经到头
                if len(ohlcv) < limit_per_request:
                    break

                print(f"Fetched {len(all_data)} candles...")

            except Exception as e:
                print(f"Error: {e}")
                break

        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df = df.drop_duplicates()
            df = df.sort_index()
            return df
        return None


def load_data(filepath):
    """从CSV文件加载数据"""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    return None


def save_data(df, filepath):
    """保存数据到CSV文件"""
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")


def generate_mock_btc_data(n_candles=3000, base_price=45000):
    """
    生成模拟BTC数据用于测试
    基于随机游走和波动率特征
    """
    np.random.seed(42)

    # 生成时间索引
    start_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_date, periods=n_candles, freq='15min')

    # 模拟价格波动 - 使用几何布朗运动
    dt = 1/(24*4)  # 15分钟
    mu = 0.0001    # 漂移率 (稍微上涨趋势)
    sigma = 0.02   # 波动率

    returns = np.random.normal(mu, sigma, n_candles)
    log_prices = np.log(base_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # 生成OHLC
    data = []
    for i, close in enumerate(close_prices):
        # 基于收盘价生成其他价格
        volatility = sigma * close
        high = close + abs(np.random.normal(0, volatility/2))
        low = close - abs(np.random.normal(0, volatility/2))
        open_price = np.random.uniform(low, high)

        # 确保OHLC关系正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 成交量 - 与价格变动相关
        base_volume = 500
        volume = base_volume * np.random.uniform(0.5, 2) * (1 + abs(returns[i]) * 10)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'datetime'

    print(f"Generated {n_candles} mock BTC candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df

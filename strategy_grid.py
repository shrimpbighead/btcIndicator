"""
BTC 15分钟 - 快速网格搜索版
简化参数，快速找到高胜率配置
"""
import os
import pandas as pd
import numpy as np

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import TRAIN_TEST_SPLIT


def get_data():
    data_file = "data/BTC_USDT_15m.csv"
    df = load_data(data_file)
    if df is None:
        try:
            fetcher = BinanceDataFetcher("BTC/USDT", "15m", proxy=DEFAULT_PROXY)
            df = fetcher.fetch_historical_data(days=400)
            if df is not None:
                save_data(df, data_file)
        except:
            pass
    return df


def add_indicators(df):
    # MA
    for p in [5, 10, 20]:
        df[f'ma{p}'] = df['close'].rolling(p).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    # Stochastic
    low = df['low'].rolling(14).min()
    high = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low) / (high - low + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    return df


def test_config(df, rsi_buy, rsi_sell, macd_enable, stoch_enable, bb_enable, threshold):
    """测试单个配置"""
    df = add_indicators(df.copy())

    scores = pd.Series(0.0, index=df.index)

    # RSI
    scores[df['rsi'] < rsi_buy] += 1
    scores[df['rsi'] > rsi_sell] -= 1

    # MACD金叉死叉
    if macd_enable:
        buy = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
        sell = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
        scores[buy] += 1
        scores[sell] -= 1

    # Stochastic
    if stoch_enable:
        buy = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        sell = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        scores[buy] += 1
        scores[sell] -= 1

    # BB
    if bb_enable:
        scores[df['close'] < df['bb_lower']] += 1

    # 信号
    signals = pd.Series(0, index=df.index)
    signals[scores >= threshold] = 1
    signals[scores <= -threshold] = -1

    # 计算胜率（不扣手续费）
    wins = 0
    total = 0

    for i in range(1, len(df)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry = df.iloc[i]['close']
        exit_p = df.iloc[i+1]['close']

        if sig == 1:  # 做多
            correct = exit_p > entry
        else:
            correct = exit_p < entry

        if correct:
            wins += 1
        total += 1

    if total == 0:
        return 0, 0

    return wins / total, total


def main():
    print("=" * 60)
    print("BTC 15min - Grid Search")
    print("=" * 60)

    df = get_data()
    if df is None:
        print("No data")
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 网格搜索
    best = None
    best_params = None

    print("\nSearching...")

    for rsi_buy in [20, 25, 30, 35]:
        for rsi_sell in [65, 70, 75, 80]:
            for macd in [0, 1]:
                for stoch in [0, 1]:
                    for bb in [0, 1]:
                        for thresh in [1, 2, 3]:
                            wr, trades = test_config(
                                train_df, rsi_buy, rsi_sell,
                                macd, stoch, bb, thresh
                            )

                            if trades >= 50:
                                print(f"RSI({rsi_buy},{rsi_sell}) MACD={macd} STOCH={stoch} BB={bb} T={thresh}: "
                                      f"WR={wr:.2%}, Trades={trades}")

                                if best is None or wr > best:
                                    best = wr
                                    best_params = {
                                        'rsi_buy': rsi_buy,
                                        'rsi_sell': rsi_sell,
                                        'macd': macd,
                                        'stoch': stoch,
                                        'bb': bb,
                                        'threshold': thresh
                                    }

    print("\n" + "=" * 60)
    print("BEST TRAIN:", best_params, best)

    # 测试集
    if best_params:
        wr, trades = test_config(
            test_df,
            best_params['rsi_buy'],
            best_params['rsi_sell'],
            best_params['macd'],
            best_params['stoch'],
            best_params['bb'],
            best_params['threshold']
        )

        print("\nTEST RESULT:")
        print(f"Win Rate: {wr:.2%}")
        print(f"Trades: {trades}")

        if wr >= 0.60:
            print("\n*** TARGET ACHIEVED! ***")


if __name__ == "__main__":
    main()

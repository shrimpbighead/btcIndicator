"""
BTC 15分钟 - 参数优化全量版
"""
import pandas as pd
import numpy as np

from data.fetcher import load_data


def add_indicators(df):
    df['rsi'] = 100 - (100 / (1 +
        df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() /
        (df['close'].diff().where(df['close'].diff() < 0, 0).rolling(14).mean() + 0.0001)))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    return df


def test(rsi_buy, rsi_sell, macd, threshold):
    df = add_indicators(load_data("data/BTC_USDT_15m.csv").copy())

    scores = pd.Series(0.0, index=df.index)
    scores[df['rsi'] < rsi_buy] += 1
    scores[df['rsi'] > rsi_sell] -= 1
    if macd:
        scores[(df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))] += 1
        scores[(df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))] -= 1

    signals = pd.Series(0, index=df.index)
    signals[scores >= threshold] = 1

    wins = 0
    total = 0
    for i in range(1, len(df)-1):
        if signals.iloc[i] == 1:
            if df.iloc[i+1]['close'] > df.iloc[i]['close']:
                wins += 1
            total += 1

    return wins/total*100 if total > 0 else 0, total


print("优化参数...")
best = 0
best_params = None

for rsi_buy in [25, 30, 35, 40]:
    for rsi_sell in [65, 70, 75, 80]:
        for macd in [0, 1]:
            for thresh in [2, 3]:
                wr, trades = test(rsi_buy, rsi_sell, macd, thresh)
                if trades >= 30:
                    print(f"RSI({rsi_buy},{rsi_sell}) MACD={macd} T={thresh}: {wr:.2f}% ({trades})")
                    if wr > best:
                        best = wr
                        best_params = (rsi_buy, rsi_sell, macd, thresh)

print(f"\n最佳: {best_params} -> {best:.2f}%")

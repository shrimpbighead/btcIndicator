"""
BTC 15分钟策略 - 全量数据分析
使用全部历史数据测试
"""
import pandas as pd
import numpy as np
import json

from data.fetcher import load_data


def add_indicators(df):
    for p in [5, 10, 20]:
        df[f'ma{p}'] = df['close'].rolling(p).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    low = df['low'].rolling(14).min()
    high = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low) / (high - low + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    return df


def run_full_test():
    df = load_data("data/BTC_USDT_15m.csv")

    print("=" * 70)
    print("全量数据测试")
    print("=" * 70)

    print(f"\n总K线: {len(df)}")
    print(f"时间: {df.index[0]} 到 {df.index[-1]}")
    print(f"跨度: {(df.index[-1] - df.index[0]).days} 天")

    # 最佳参数
    params = {
        'rsi_buy': 35,
        'rsi_sell': 80,
        'macd': 1,
        'threshold': 2
    }

    df = add_indicators(df.copy())

    scores = pd.Series(0.0, index=df.index)
    scores[df['rsi'] < params['rsi_buy']] += 1
    scores[df['rsi'] > params['rsi_sell']] -= 1

    buy = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
    sell = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
    scores[buy] += 1
    scores[sell] -= 1

    signals = pd.Series(0, index=df.index)
    signals[scores >= params['threshold']] = 1
    signals[scores <= -params['threshold']] = -1

    # 全部数据的交易
    trades = []
    for i in range(1, len(df)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        trades.append({
            'time': str(df.index[i])[:19],
            'direction': 'LONG' if sig == 1 else 'SHORT',
            'entry': df.iloc[i]['close'],
            'exit': df.iloc[i+1]['close'],
            'success': df.iloc[i+1]['close'] > df.iloc[i]['close'] if sig == 1 else df.iloc[i+1]['close'] < df.iloc[i]['close']
        })

    total = len(trades)
    success = sum(1 for t in trades if t['success'])

    print(f"\n总交易: {total}")
    print(f"成功: {success}")
    print(f"失败: {total - success}")
    print(f"胜率: {success/total*100:.2f}%")

    # 保存
    with open('results/all_trades.json', 'w') as f:
        json.dump(trades, f, indent=2)


if __name__ == "__main__":
    run_full_test()

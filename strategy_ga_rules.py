"""
BTC 15分钟 - 规则策略 + GA优化
用技术指标规则生成信号，然后用GA找最优参数
"""
import os
import pandas as pd
import numpy as np
import random

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


def calculate_indicators(df):
    """计算技术指标"""
    result = df.copy()

    # MA
    for p in [5, 10, 20, 40]:
        result[f'ma{p}'] = df['close'].rolling(p).mean()

    # EMA
    for p in [5, 10, 20]:
        result[f'ema{p}'] = df['close'].ewm(span=p).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    result['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    result['macd'] = ema12 - ema26
    result['macd_sig'] = result['macd'].ewm(span=9).mean()
    result['macd_hist'] = result['macd'] - result['macd_sig']

    # 布林带
    result['bb_mid'] = df['close'].rolling(20).mean()
    result['bb_std'] = df['close'].rolling(20).std()
    result['bb_upper'] = result['bb_mid'] + 2 * result['bb_std']
    result['bb_lower'] = result['bb_mid'] - 2 * result['bb_std']

    return result


def generate_signals(df, params):
    """根据参数生成信号"""
    df = calculate_indicators(df)

    signals = pd.Series(0, index=df.index)
    scores = pd.Series(0, index=df.index)

    # 规则池
    # 1. MA金叉
    if params.get('ma_cross', 0) == 1:
        for p1, p2 in [(5, 10), (5, 20), (10, 20)]:
            ma1, ma2 = df[f'ma{p1}'], df[f'ma{p2}']
            # 金叉买入
            buy = (df['close'] > ma1) & (df['close'].shift(1) <= ma1.shift(1)) & (ma1 > ma2)
            # 死叉卖出
            sell = (df['close'] < ma1) & (df['close'].shift(1) >= ma1.shift(1)) & (ma1 < ma2)
            scores[buy] += 1
            scores[sell] -= 1

    # 2. RSI超卖/超买
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    buy = df['rsi'] < rsi_oversold
    sell = df['rsi'] > rsi_overbought
    scores[buy] += 1
    scores[sell] -= 1

    # 3. MACD金叉/死叉
    if params.get('macd_cross', 0) == 1:
        buy = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
        sell = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
        scores[buy] += 1
        scores[sell] -= 1

    # 4. 布林带突破
    if params.get('bb_break', 0) == 1:
        buy = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        sell = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        scores[buy] += 1
        scores[sell] -= 1

    # 5. 均线多头/空头排列
    if params.get('ma_trend', 0) == 1:
        # 多头排列买入
        buy = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']) & (df['close'] > df['ma5'])
        # 空头排列卖出
        sell = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']) & (df['close'] < df['ma5'])
        scores[buy] += 1
        scores[sell] -= 1

    # 6. 连续下跌/上涨
    if params.get('consecutive', 0) == 1:
        n = params.get('consecutive_n', 3)
        ret = df['close'].pct_change()
        buy = (ret.shift(1) < -0.002) & (ret.shift(2) < -0.002)
        sell = (ret.shift(1) > 0.002) & (ret.shift(2) > 0.002)
        scores[buy] += 1
        scores[sell] -= 1

    # 信号阈值
    threshold = params.get('threshold', 1)
    signals[scores >= threshold] = 1
    signals[scores <= -threshold] = -1

    return signals


def evaluate(params, test_df):
    """评估参数"""
    signals = generate_signals(test_df, params)

    # 回测：下一根K线平仓
    trades = []
    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry = test_df['close'].iloc[i]
        exit_p = test_df['close'].iloc[i+1]

        if sig == 1:
            ret = (exit_p - entry) / entry - 0.002
        else:
            ret = (entry - exit_p) / entry - 0.002

        trades.append(ret)

    if len(trades) < 10:
        return {'win_rate': 0, 'trades': 0, 'return': 0}

    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades)
    total_ret = sum(trades)

    return {'win_rate': win_rate, 'trades': len(trades), 'return': total_ret}


def genetic_algorithm(train_df, generations=20):
    """遗传算法优化"""
    print("GA Optimization...")

    # 参数空间
    param_space = {
        'ma_cross': [0, 1],
        'rsi_oversold': [20, 25, 30, 35],
        'rsi_overbought': [65, 70, 75, 80],
        'macd_cross': [0, 1],
        'bb_break': [0, 1],
        'ma_trend': [0, 1],
        'consecutive': [0, 1],
        'consecutive_n': [2, 3, 4],
        'threshold': [1, 2, 3]
    }

    # 初始化种群
    population = []
    for _ in range(50):
        params = {k: random.choice(v) for k, v in param_space.items()}
        population.append(params)

    best_result = None
    best_params = None

    for gen in range(generations):
        # 评估
        results = []
        for params in population:
            result = evaluate(params, train_df)
            result['params'] = params
            results.append(result)

        # 排序
        results.sort(key=lambda x: x['win_rate'], reverse=True)

        # 打印最好的
        best = results[0]
        print(f"Gen {gen}: Best WR={best['win_rate']:.2%}, Trades={best['trades']}")

        if best_result is None or best['win_rate'] > best_result['win_rate']:
            best_result = best
            best_params = best['params']

        # 选择
        elite = [r['params'] for r in results[:10]]

        # 生成下一代
        new_population = elite.copy()
        while len(new_population) < 50:
            parent = random.choice(elite)
            child = parent.copy()

            # 变异
            for key in child:
                if random.random() < 0.3:
                    child[key] = random.choice(param_space[key])

            new_population.append(child)

        population = new_population

    return best_params, best_result


def main():
    print("=" * 60)
    print("BTC 15min - Rule-based Strategy with GA")
    print("=" * 60)

    df = get_data()
    if df is None:
        print("No data")
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # GA优化
    best_params, train_result = genetic_algorithm(train_df)

    print("\n" + "=" * 60)
    print("Best params:", best_params)
    print("Train result:", train_result)

    # 测试集验证
    test_result = evaluate(best_params, test_df)

    print("\n" + "=" * 60)
    print("TEST RESULT")
    print("=" * 60)
    print(f"Win Rate: {test_result['win_rate']:.2%}")
    print(f"Trades: {test_result['trades']}")
    print(f"Return: {test_result['return']:.2%}")

    if test_result['win_rate'] >= 0.60:
        print("\n*** TARGET ACHIEVED! ***")


if __name__ == "__main__":
    main()

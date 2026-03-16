"""
BTC 15分钟策略 - 纯胜率版
不考虑手续费，只追求胜率
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

    # Stochastic
    low = df['low'].rolling(14).min()
    high = df['high'].rolling(14).max()
    result['stoch_k'] = 100 * (df['close'] - low) / (high - low + 0.0001)
    result['stoch_d'] = result['stoch_k'].rolling(3).mean()

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['atr'] = tr.rolling(14).mean()

    return result


def generate_signals(df, params):
    df = calculate_indicators(df)
    signals = pd.Series(0, index=df.index)
    scores = pd.Series(0.0, index=df.index)  # 使用浮点数

    # 1. RSI超卖/超买
    rsi_buy = df['rsi'] < params.get('rsi_oversold', 30)
    rsi_sell = df['rsi'] > params.get('rsi_overbought', 70)
    scores[rsi_buy] += params.get('rsi_weight', 1)
    scores[rsi_sell] -= params.get('rsi_weight', 1)

    # 2. MACD金叉/死叉
    if params.get('macd_cross', 1):
        macd_buy = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
        macd_sell = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
        scores[macd_buy] += params.get('macd_weight', 1)
        scores[macd_sell] -= params.get('macd_weight', 1)

    # 3. Stochastic
    if params.get('stoch_cross', 1):
        stoch_buy = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        stoch_sell = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        scores[stoch_buy] += params.get('stoch_weight', 1)
        scores[stoch_sell] -= params.get('stoch_weight', 1)

    # 4. 布林带突破
    if params.get('bb_break', 1):
        bb_buy = df['close'] < df['bb_lower']
        bb_sell = df['close'] > df['bb_upper']
        scores[bb_buy] += params.get('bb_weight', 1)
        scores[bb_sell] -= params.get('bb_weight', 1)

    # 5. MA金叉
    if params.get('ma_cross', 1):
        for p1, p2 in [(5, 10), (5, 20)]:
            ma1, ma2 = df[f'ma{p1}'], df[f'ma{p2}']
            buy = (df['close'] > ma1) & (df['close'].shift(1) <= ma1.shift(1))
            sell = (df['close'] < ma1) & (df['close'].shift(1) >= ma1.shift(1))
            scores[buy] += 0.5
            scores[sell] -= 0.5

    # 6. 连续下跌/上涨
    if params.get('consecutive', 1):
        n = params.get('consecutive_n', 2)
        for i in range(1, n+1):
            scores[df['close'].pct_change(i) < -0.001] += 0.3
            scores[df['close'].pct_change(i) > 0.001] -= 0.3

    # 7. 价格突破均线
    if params.get('price_ma', 1):
        for p in [10, 20]:
            ma = df[f'ma{p}']
            buy = (df['close'] > ma) & (df['close'].shift(1) <= ma.shift(1))
            sell = (df['close'] < ma) & (df['close'].shift(1) >= ma.shift(1))
            scores[buy] += 0.5
            scores[sell] -= 0.5

    # 8. ATR突破
    if params.get('atr_break', 1):
        atr_mult = params.get('atr_mult', 1.5)
        buy = df['close'] < df['close'].shift(1) - atr_mult * df['atr']
        sell = df['close'] > df['close'].shift(1) + atr_mult * df['atr']
        scores[buy] += params.get('atr_weight', 1)
        scores[sell] -= params.get('atr_weight', 1)

    # 信号阈值
    threshold = params.get('threshold', 1)
    signals[scores >= threshold] = 1
    signals[scores <= -threshold] = -1

    return signals


def evaluate(params, test_df):
    """评估 - 不扣手续费"""
    signals = generate_signals(params=test_df, df=params['_df'])

    # 回测：下一根K线平仓
    wins = 0
    total = 0

    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry = params['_df'].iloc[i]['close']
        exit_p = params['_df'].iloc[i+1]['close']

        if sig == 1:  # 做多
            correct = exit_p > entry
        else:  # 做空
            correct = exit_p < entry

        if correct:
            wins += 1
        total += 1

    if total == 0:
        return {'win_rate': 0, 'trades': 0}

    return {'win_rate': wins / total, 'trades': total}


def run_ga():
    print("=" * 60)
    print("BTC 15min - Pure Win Rate (No Fees)")
    print("=" * 60)

    df = get_data()
    if df is None:
        print("No data")
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 参数空间
    param_space = {
        'rsi_oversold': [20, 25, 30, 35],
        'rsi_overbought': [65, 70, 75, 80],
        'rsi_weight': [1, 2, 3],
        'macd_cross': [0, 1],
        'macd_weight': [1, 2, 3],
        'stoch_cross': [0, 1],
        'stoch_weight': [1, 2, 3],
        'bb_break': [0, 1],
        'bb_weight': [1, 2, 3],
        'ma_cross': [0, 1],
        'consecutive': [0, 1],
        'consecutive_n': [2, 3],
        'price_ma': [0, 1],
        'atr_break': [0, 1],
        'atr_weight': [1, 2, 3],
        'atr_mult': [1, 1.5, 2],
        'threshold': [1, 2, 3, 4]
    }

    # 初始化
    population = []
    for _ in range(100):
        p = {k: random.choice(v) for k, v in param_space.items()}
        p['_df'] = train_df
        population.append(p)

    best_result = None
    best_params = None

    for gen in range(30):
        # 评估
        results = []
        for params in population:
            result = evaluate(params, train_df)
            result['params'] = {k: v for k, v in params.items() if k != '_df'}
            results.append(result)

        # 排序
        results.sort(key=lambda x: x['win_rate'], reverse=True)

        best = results[0]
        print(f"Gen {gen}: WR={best['win_rate']:.2%}, Trades={best['trades']}")

        if best_result is None or best['win_rate'] > best_result['win_rate']:
            best_result = best
            best_params = {k: v for k, v in best['params'].items()}

        # 精英保留
        elite = [r['params'] for r in results[:20]]

        # 生成下一代
        new_pop = [{**p, '_df': train_df} for p in elite[:10]]
        while len(new_pop) < 100:
            parent = random.choice(elite)
            child = {**parent, '_df': train_df}
            # 变异
            for key in child:
                if key != '_df' and random.random() < 0.2:
                    child[key] = random.choice(param_space[key])
            new_pop.append(child)

        population = new_pop

    print("\n" + "=" * 60)
    print("BEST PARAMS:", best_params)
    print("TRAIN RESULT:", best_result)

    # 测试集
    test_params = {**best_params, '_df': test_df}
    test_result = evaluate(test_params, test_df)

    print("\n" + "=" * 60)
    print("TEST RESULT")
    print("=" * 60)
    print(f"Win Rate: {test_result['win_rate']:.2%}")
    print(f"Trades: {test_result['trades']}")

    if test_result['win_rate'] >= 0.60:
        print("\n*** TARGET ACHIEVED! ***")


if __name__ == "__main__":
    run_ga()

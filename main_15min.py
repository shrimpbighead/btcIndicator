"""
BTC 15分钟级别策略 - 真正15分钟交易版
每根K线只持有15分钟，开仓后下一根K线立即平仓
"""
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT


def get_data():
    """加载数据"""
    data_file = f"data/BTC_USDT_15m.csv"

    df = load_data(data_file)
    if df is not None:
        return df

    print("Fetching data...")
    try:
        fetcher = BinanceDataFetcher("BTC/USDT", "15m", proxy=DEFAULT_PROXY)
        df = fetcher.fetch_historical_data(days=400)
        if df is not None:
            save_data(df, data_file)
    except:
        pass
    return df


def create_features(df):
    """创建特征 - 用于预测下一根K线方向"""
    features = pd.DataFrame(index=df.index)

    # 价格变化 - 核心特征
    for p in [1, 2, 3, 4, 5]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 移动平均
    for p in [5, 10, 20]:
        ma = df['close'].rolling(p).mean()
        features[f'pma{p}'] = (df['close'] - ma) / ma

    # 波动率
    features['vol'] = df['close'].pct_change().rolling(8).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    features['macd'] = macd
    features['macd_sig'] = macd.ewm(span=9).mean()

    # 布林带位置
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    features['bb_pos'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 0.0001)

    # 成交量变化
    features['vch'] = df['volume'].pct_change()

    # 标签：下一根K线是涨还是跌 (不是未来4根，是1根！)
    next_bar_return = df['close'].shift(-1) / df['close'] - 1
    labels = (next_bar_return > 0).astype(int)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.fillna(0).astype(int)

    # 有效数据
    valid = features.index[200:-1]  # 最后一行不要，因为没有下一根K线
    return features.loc[valid], labels.loc[valid]


def run_15min_strategy():
    """运行真正的15分钟策略"""
    print("=" * 60)
    print("BTC 15m Strategy - 15分钟持有")
    print("=" * 60)

    df = get_data()
    if df is None:
        print("No data")
        return

    # 分割
    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 特征
    X, y = create_features(df)
    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 训练
    print("\nTraining...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    # 测试集准确率
    test_pred = model.predict(X_test_sc)
    test_acc = (test_pred == y_test).mean()
    print(f"Test accuracy: {test_acc:.2%}")

    # 回测 - 关键：每根K线只持有15分钟
    # 在当前K线收盘时开仓，下一根K线收盘时平仓
    proba = model.predict_proba(X_test_sc)[:, 1]

    # 只做多，不做空
    threshold = 0.52  # 稍微降低阈值，增加交易频率

    signals = pd.Series(0, index=X_test.index)
    signals[proba >= threshold] = 1  # 买入信号
    # 不做空，只在预测下跌时不持仓

    print(f"\nSignals: {(signals==1).sum()}")

    # 15分钟级别回测
    # 信号在K线i产生，在K线i+1收盘时平仓
    test_prices = test_df.loc[X_test.index]

    trades = []
    for i in range(len(signals)-1):
        if signals.iloc[i] == 1:
            # 开仓价格：当前K线收盘价
            entry_price = test_prices['close'].iloc[i]
            # 平仓价格：下一根K线收盘价
            exit_price = test_prices['close'].iloc[i+1]

            # 收益率
            ret = (exit_price - entry_price) / entry_price
            ret -= 0.002  # 手续费

            trades.append({
                'entry_time': str(test_prices.index[i]),
                'exit_time': str(test_prices.index[i+1]),
                'entry': entry_price,
                'exit': exit_price,
                'return': ret
            })

    # 结果
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (15min持有)")
    print("=" * 60)

    if trades:
        trades_df = pd.DataFrame(trades)
        wins = (trades_df['return'] > 0).sum()
        win_rate = wins / len(trades)
        total_return = trades_df['return'].sum()

        print(f"\nTotal Trades: {len(trades)} (每笔持有15分钟)")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_return:.2%}")

        # 平均每笔收益
        avg_ret = trades_df['return'].mean()
        print(f"Avg Return per trade: {avg_ret:.4%}")

        # 统计
        print(f"\nWins: {wins}")
        print(f"Losses: {len(trades) - wins}")

        if win_rate >= 0.60:
            print(f"\n*** TARGET ACHIEVED! ***")

        # 保存
        os.makedirs('results', exist_ok=True)
        with open('results/trades_15min.json', 'w') as f:
            json.dump(trades, f, indent=2)

        print("\nResults saved")

    return trades if trades else []


# 也测试做空
def run_15min_both():
    """双向交易版本"""
    print("\n" + "=" * 60)
    print("BTC 15m Strategy - 双向交易")
    print("=" * 60)

    df = get_data()
    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X, y = create_features(df)
    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=30, min_samples_leaf=15,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    proba = model.predict_proba(X_test_sc)[:, 1]
    test_acc = (model.predict(X_test_sc) == y_test).mean()
    print(f"Test accuracy: {test_acc:.2%}")

    # 双向信号
    threshold = 0.55
    signals = pd.Series(0, index=X_test.index)
    signals[proba >= threshold] = 1   # 做多
    signals[proba <= (1-threshold)] = -1  # 做空

    print(f"Buy: {(signals==1).sum()}, Sell: {(signals==-1).sum()}")

    # 回测
    test_prices = test_df.loc[X_test.index]
    trades = []

    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry = test_prices['close'].iloc[i]
        exit_p = test_prices['close'].iloc[i+1]

        if sig == 1:  # 做多
            ret = (exit_p - entry) / entry
        else:  # 做空
            ret = (entry - exit_p) / entry

        ret -= 0.002

        trades.append({
            'entry': entry,
            'exit': exit_p,
            'direction': 'LONG' if sig == 1 else 'SHORT',
            'return': ret
        })

    if trades:
        trades_df = pd.DataFrame(trades)
        wins = (trades_df['return'] > 0).sum()
        win_rate = wins / len(trades)
        total_ret = trades_df['return'].sum()

        print(f"\nTrades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_ret:.2%}")

        if win_rate >= 0.60:
            print("*** TARGET ACHIEVED! ***")


if __name__ == "__main__":
    # 只做多
    run_15min_strategy()

    # 双向
    run_15min_both()

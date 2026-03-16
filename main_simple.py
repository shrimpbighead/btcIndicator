"""
BTC 15分钟级别策略 - 改进版
使用高置信度信号的ML策略
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT


def load_or_fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=400):
    """加载或获取数据"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"

    df = load_data(data_file)
    if df is not None:
        print(f"Loaded: {len(df)} candles")
        return df

    print("Fetching data...")
    try:
        fetcher = BinanceDataFetcher(symbol, timeframe, proxy=DEFAULT_PROXY)
        df = fetcher.fetch_historical_data(days=days)
        if df is not None:
            save_data(df, data_file)
            return df
    except Exception as e:
        print(f"Failed: {e}")
    return None


def create_features(df, look_ahead=4):
    """创建特征"""
    features = pd.DataFrame(index=df.index)

    # 价格变化
    for p in [1, 2, 3, 4, 5, 8, 12, 24]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 移动平均
    for p in [5, 10, 20, 40]:
        ma = df['close'].rolling(p).mean()
        features[f'pma{p}'] = (df['close'] - ma) / ma

    # 波动率
    for p in [8, 16, 24]:
        features[f'vol{p}'] = df['close'].pct_change().rolling(p).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    features['macd'] = macd
    features['macd_sig'] = macd.ewm(span=9).mean()

    # 成交量
    features['vch'] = df['volume'].pct_change()
    features['vrat'] = df['volume'] / df['volume'].rolling(20).mean()

    # 标签
    future_ret = df['close'].shift(-look_ahead) / df['close'] - 1
    labels = (future_ret > 0).astype(int)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.fillna(0).astype(int)

    # 有效范围
    valid = features.index[200:-look_ahead]
    return features.loc[valid], labels.loc[valid]


def run_strategy():
    """运行策略"""
    print("=" * 50)
    print("BTC 15m ML Strategy")
    print("=" * 50)

    df = load_or_fetch_data()
    if df is None:
        return

    # 分割
    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 特征
    X, y = create_features(df, look_ahead=4)
    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

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

    # 预测
    pred = model.predict(X_test_sc)
    proba = model.predict_proba(X_test_sc)[:, 1]

    train_acc = (model.predict(X_train_sc) == y_train).mean()
    test_acc = (pred == y_test).mean()

    print(f"Train acc: {train_acc:.2%}")
    print(f"Test acc: {test_acc:.2%}")

    # 高置信度信号
    THRESHOLD = 0.58  # 高置信度阈值

    # 只在预测概率高时买入，低时卖出
    signals = pd.Series(0, index=X_test.index)
    signals[proba >= THRESHOLD] = 1   # 强烈看多
    signals[proba <= (1-THRESHOLD)] = -1  # 强烈看空

    print(f"\nSignals: {(signals==1).sum()} buy, {(signals==-1).sum()} sell")

    # 回测 - 简化版
    test_prices = test_df.loc[signals.index]

    trades = []
    pos = 0
    entry = 0

    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig != 0 and pos == 0:
            pos = sig
            entry = test_prices['close'].iloc[i]
        elif sig != 0 and pos != 0 and sig != pos:
            # 反向信号，平仓
            exit_p = test_prices['close'].iloc[i]
            ret = (exit_p - entry) / entry if pos == 1 else (entry - exit_p) / entry
            ret -= 0.002
            trades.append(ret)
            pos = sig
            entry = test_prices['close'].iloc[i]

    # 平仓
    if pos != 0:
        exit_p = test_prices['close'].iloc[-1]
        ret = (exit_p - entry) / entry if pos == 1 else (entry - exit_p) / entry
        ret -= 0.002
        trades.append(ret)

    if trades:
        wins = [t > 0 for t in trades]
        win_rate = sum(wins) / len(trades)
        total_ret = sum(trades)

        print(f"\n=== Results ===")
        print(f"Trades: {len(trades)}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Total return: {total_ret:.2%}")

        if win_rate >= 0.60:
            print("\n*** TARGET MET! ***")
    else:
        print("No trades")


if __name__ == "__main__":
    run_strategy()

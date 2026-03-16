"""
BTC 15分钟策略 - 波动率突破版
思路：预测是否有大幅波动(>0.5%)，然后根据RSI等指标判断方向
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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


def create_features(df):
    features = pd.DataFrame(index=df.index)

    # 收益率
    for p in [1, 2, 3, 4, 5]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 移动平均
    for p in [5, 10, 20]:
        ma = df['close'].rolling(p).mean()
        features[f'pma{p}'] = (df['close'] - ma) / ma

    # 波动率
    features['vol'] = df['close'].pct_change().rolling(8).std()
    features['vol_slow'] = df['close'].pct_change().rolling(20).std()

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

    # 成交量
    features['vch'] = df['volume'].pct_change()
    features['vrat'] = df['volume'] / df['volume'].rolling(20).mean()

    # 标签：有大幅波动吗？(无论涨跌)
    next_ret = df['close'].shift(-1) / df['close'] - 1
    features['big_move'] = (abs(next_ret) > 0.005).astype(int)  # 0.5%以上

    # 标签：方向（只在大幅波动时）
    features['direction'] = (next_ret > 0).astype(int)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    valid = features.index[200:-1]
    return features.loc[valid]


def run_strategy():
    print("=" * 60)
    print("BTC 15min - Volatility Strategy")
    print("=" * 60)

    df = get_data()
    if df is None:
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X = create_features(df)
    split_idx = len(train_df) - 200

    # 分离标签
    y_move = X['big_move']  # 有大幅波动？
    y_dir = X['direction']  # 方向？
    X = X.drop(['big_move', 'direction'], axis=1)

    X_train = X.iloc[:split_idx]
    y_move_train = y_move.iloc[:split_idx]
    y_dir_train = y_dir.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_move_test = y_move.iloc[split_idx:]
    y_dir_test = y_dir.iloc[split_idx:]

    test_prices = test_df.loc[X_test.index]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Big move ratio: {y_move.mean():.2%}")

    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 训练两个模型
    print("\nTraining models...")

    # 模型1：预测是否有大幅波动
    model_move = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=20, random_state=42, n_jobs=-1
    )
    model_move.fit(X_train_sc, y_move_train)
    move_proba = model_move.predict_proba(X_test_sc)[:, 1]

    # 模型2：预测方向（只看大涨/大跌的情况）
    model_dir = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=30, random_state=42, n_jobs=-1
    )
    model_dir.fit(X_train_sc, y_dir_train)
    dir_proba = model_dir.predict_proba(X_test_sc)[:, 1]

    # 策略：只在预测到大幅波动时交易
    move_threshold = 0.55  # 预测有大幅波动
    dir_threshold = 0.52    # 预测方向

    signals = pd.Series(0, index=X_test.index)

    for i in range(len(X_test)):
        if move_proba[i] >= move_threshold:
            # 预测有大幅波动
            if dir_proba[i] >= dir_threshold:
                signals.iloc[i] = 1  # 做多
            elif dir_proba[i] <= (1 - dir_threshold):
                signals.iloc[i] = -1  # 做空

    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    print(f"\nSignals: Buy={buy_signals}, Sell={sell_signals}")

    # 回测：下一根K线平仓
    trades = []
    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry = test_prices['close'].iloc[i]
        exit_p = test_prices['close'].iloc[i+1]

        if sig == 1:  # 做多
            ret = (exit_p - entry) / entry - 0.002
        else:  # 做空
            ret = (entry - exit_p) / entry - 0.002

        trades.append(ret)

    if trades:
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades)
        total_ret = sum(trades)

        print(f"\nTrades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_ret:.2%}")

        if win_rate >= 0.60:
            print("\n*** TARGET ACHIEVED! ***")

    return trades


if __name__ == "__main__":
    run_strategy()

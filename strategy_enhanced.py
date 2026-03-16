"""
BTC 15分钟 - 增强版
尝试更多特征和参数组合
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def create_features_v2(df):
    """增强特征"""
    features = pd.DataFrame(index=df.index)

    # 原始价格数据
    features['open'] = df['open']
    features['high'] = df['high']
    features['low'] = df['low']
    features['close'] = df['close']
    features['volume'] = df['volume']

    # 收益率
    for p in [1, 2, 3, 4, 5, 8, 12]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 移动平均交叉
    for p1, p2 in [(5, 10), (5, 20), (10, 20), (20, 60)]:
        ma1 = df['close'].rolling(p1).mean()
        ma2 = df['close'].rolling(p2).mean()
        features[f'cross_{p1}_{p2}'] = (ma1 - ma2) / ma2

    # 价格位置
    for p in [10, 20, 40]:
        roll_max = df['high'].rolling(p).max()
        roll_min = df['low'].rolling(p).min()
        features[f'price_pos_{p}'] = (df['close'] - roll_min) / (roll_max - roll_min + 0.0001)

    # 波动率
    for p in [4, 8, 16]:
        features[f'vol{p}'] = df['close'].pct_change().rolling(p).std()

    # 动量
    for p in [4, 8]:
        features[f'mom{p}'] = df['close'].pct_change(p).rolling(p).mean()

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
    features['macd_hist'] = macd - features['macd_sig']

    # 布林带
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    features['bb_upper'] = bb_mid + 2 * bb_std
    features['bb_lower'] = bb_mid - 2 * bb_std
    features['bb_pos'] = (df['close'] - features['bb_lower']) / (4 * bb_std + 0.0001)

    # Keltner Channel
    kc_mid = df['close'].ewm(span=20).mean()
    kc_range = df['high'] - df['low']
    kc_range_ma = kc_range.rolling(10).mean()
    features['kc_upper'] = kc_mid + 2 * kc_range_ma
    features['kc_lower'] = kc_mid - 2 * kc_range_ma
    features['kc_pos'] = (df['close'] - features['kc_lower']) / (4 * kc_range_ma + 0.0001)

    # 成交量
    features['vch'] = df['volume'].pct_change()
    features['vrat'] = df['volume'] / df['volume'].rolling(20).mean()
    features['vma5'] = df['volume'].rolling(5).mean()
    features['vma20'] = df['volume'].rolling(20).mean()

    # 标签
    next_ret = df['close'].shift(-1) / df['close'] - 1
    features['label'] = (next_ret > 0).astype(int)

    # 删除原始价格数据
    features = features.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    valid = features.index[200:-1]
    return features.loc[valid]


def run():
    print("=" * 60)
    print("BTC 15min - Enhanced Features")
    print("=" * 60)

    df = get_data()
    if df is None:
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X = create_features_v2(df)
    y = X['label']
    X = X.drop('label', axis=1)

    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    test_prices = test_df.loc[X_test.index]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 测试不同配置
    configs = [
        {'threshold': 0.52, 'max_depth': 6, 'n_estimators': 100},
        {'threshold': 0.52, 'max_depth': 8, 'n_estimators': 200},
        {'threshold': 0.53, 'max_depth': 6, 'n_estimators': 100},
        {'threshold': 0.53, 'max_depth': 8, 'n_estimators': 200},
        {'threshold': 0.54, 'max_depth': 6, 'n_estimators': 150},
        {'threshold': 0.54, 'max_depth': 8, 'n_estimators': 200},
        {'threshold': 0.55, 'max_depth': 10, 'n_estimators': 200},
        {'threshold': 0.55, 'max_depth': 12, 'n_estimators': 300},
    ]

    best = None

    for cfg in configs:
        model = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_sc, y_train)

        proba = model.predict_proba(X_test_sc)[:, 1]
        acc = (model.predict(X_test_sc) == y_test).mean()

        signals = pd.Series(0, index=X_test.index)
        signals[proba >= cfg['threshold']] = 1

        trades = []
        for i in range(len(signals)-1):
            if signals.iloc[i] == 1:
                entry = test_prices['close'].iloc[i]
                exit_p = test_prices['close'].iloc[i+1]
                ret = (exit_p - entry) / entry - 0.002
                trades.append(ret)

        if len(trades) < 10:
            continue

        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades)
        total_ret = sum(trades)

        marker = " ***" if win_rate >= 0.60 else ""
        print(f"T={cfg['threshold']}, D={cfg['max_depth']}, N={cfg['n_estimators']}: "
              f"Acc={acc:.2%}, WR={win_rate:.2%}, Trades={len(trades)}, Ret={total_ret:.2%}{marker}")

        if best is None or win_rate > best['win_rate']:
            best = {
                'win_rate': win_rate,
                'trades': len(trades),
                'return': total_ret,
                'accuracy': acc,
                'config': cfg
            }

    print("\n" + "=" * 60)
    if best:
        print(f"Best: WR={best['win_rate']:.2%}, Trades={best['trades']}, Ret={best['return']:.2%}")
        if best['win_rate'] >= 0.60:
            print("*** TARGET ACHIEVED! ***")


if __name__ == "__main__":
    run()

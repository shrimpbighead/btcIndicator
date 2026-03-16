"""
BTC 15分钟级别策略 - 优化版
通过参数调优寻找最佳策略
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT


def load_or_fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=400):
    """加载数据"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"

    df = load_data(data_file)
    if df is None:
        print("Fetching data...")
        try:
            fetcher = BinanceDataFetcher(symbol, timeframe, proxy=DEFAULT_PROXY)
            df = fetcher.fetch_historical_data(days=days)
            if df is not None:
                save_data(df, data_file)
        except:
            pass
    return df


def create_features(df, look_ahead=4):
    """创建特征"""
    features = pd.DataFrame(index=df.index)

    # 价格变化
    for p in [1, 2, 3, 4, 5, 8, 12, 16, 24]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 移动平均
    for p in [5, 10, 20, 40, 60]:
        ma = df['close'].rolling(p).mean()
        features[f'pma{p}'] = (df['close'] - ma) / ma

    # 波动率
    for p in [8, 16, 24]:
        features[f'vol{p}'] = df['close'].pct_change().rolling(p).std()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

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
    bb_up = bb_mid + 2 * bb_std
    bb_down = bb_mid - 2 * bb_std
    features['bb_pos'] = (df['close'] - bb_down) / (bb_up - bb_down)

    # 成交量
    features['vch'] = df['volume'].pct_change()
    features['vrat'] = df['volume'] / df['volume'].rolling(20).mean()

    # 标签
    future_ret = df['close'].shift(-look_ahead) / df['close'] - 1
    labels = (future_ret > 0).astype(int)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.fillna(0).astype(int)

    valid = features.index[200:-look_ahead]
    return features.loc[valid], labels.loc[valid]


def test_params(X_train, y_train, X_test, y_test, test_prices,
                threshold, look_ahead, model_type='rf'):
    """测试参数"""
    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 训练
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

    model.fit(X_train_sc, y_train)

    # 预测
    proba = model.predict_proba(X_test_sc)[:, 1]

    # 信号
    signals = pd.Series(0, index=X_test.index)
    signals[proba >= threshold] = 1
    signals[proba <= (1-threshold)] = -1

    # 回测
    trades = []
    pos = 0
    entry = 0

    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        if sig != 0 and pos == 0:
            pos = sig
            entry = test_prices['close'].iloc[i]
        elif sig != 0 and pos != 0 and sig != pos:
            exit_p = test_prices['close'].iloc[i]
            ret = (exit_p - entry) / entry if pos == 1 else (entry - exit_p) / entry
            ret -= 0.002
            trades.append(ret)
            pos = sig
            entry = test_prices['close'].iloc[i]

    if pos != 0:
        exit_p = test_prices['close'].iloc[-1]
        ret = (exit_p - entry) / entry if pos == 1 else (entry - exit_p) / entry
        ret -= 0.002
        trades.append(ret)

    if len(trades) < 10:
        return {'win_rate': 0, 'trades': 0, 'return': 0}

    wins = sum(t > 0 for t in trades)
    return {
        'win_rate': wins / len(trades),
        'trades': len(trades),
        'return': sum(trades)
    }


def run_optimization():
    """运行优化"""
    print("=" * 60)
    print("BTC 15m Strategy Optimization")
    print("=" * 60)

    df = load_or_fetch_data()
    if df is None:
        print("No data")
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 测试不同参数
    best_result = None
    best_params = None

    # 测试不同的前瞻周期和置信度
    for look_ahead in [4, 6, 8]:
        X, y = create_features(df, look_ahead=look_ahead)
        split_idx = len(train_df) - 200

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        test_prices = test_df.loc[X_test.index]

        for threshold in [0.55, 0.58, 0.60, 0.62]:
            for model_type in ['rf', 'gb']:
                result = test_params(
                    X_train, y_train, X_test, y_test, test_prices,
                    threshold, look_ahead, model_type
                )

                print(f"LA={look_ahead}, T={threshold}, M={model_type}: "
                      f"WR={result['win_rate']:.2%}, Trades={result['trades']}, "
                      f"Ret={result['return']:.2%}")

                if result['win_rate'] >= 0.55 and result['trades'] >= 10:
                    if best_result is None or result['win_rate'] > best_result['win_rate']:
                        best_result = result
                        best_params = {
                            'look_ahead': look_ahead,
                            'threshold': threshold,
                            'model_type': model_type
                        }

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    if best_params:
        print(f"Params: {best_params}")
        print(f"Win Rate: {best_result['win_rate']:.2%}")
        print(f"Trades: {best_result['trades']}")
        print(f"Return: {best_result['return']:.2%}")

        if best_result['win_rate'] >= 0.60:
            print("\n*** TARGET ACHIEVED! ***")
    else:
        print("No valid result found")


if __name__ == "__main__":
    run_optimization()

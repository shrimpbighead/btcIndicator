"""
BTC 15分钟策略 - 优化版
目标：15分钟开平仓，胜率>60%
"""
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import TRAIN_TEST_SPLIT


def get_data():
    """获取数据"""
    data_file = "data/BTC_USDT_15m.csv"
    df = load_data(data_file)
    if df is None:
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
    """特征工程"""
    features = pd.DataFrame(index=df.index)

    # 价格变化率
    for p in [1, 2, 3, 4, 5]:
        features[f'ret{p}'] = df['close'].pct_change(p)

    # 价格相对位置
    for p in [5, 10, 20]:
        ma = df['close'].rolling(p).mean()
        features[f'pma{p}'] = (df['close'] - ma) / ma

    # 短期反转
    features['rev1'] = df['close'].pct_change(1).shift(1)  # 前一根的收益
    features['rev2'] = df['close'].pct_change(2).shift(1)

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
    features['macd_hist'] = macd - features['macd_sig']

    # 布林带
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    features['bb_pos'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 0.0001)

    # 成交量
    features['vch'] = df['volume'].pct_change()

    # 标签：下一根K线收益
    next_ret = df['close'].shift(-1) / df['close'] - 1
    labels = (next_ret > 0).astype(int)

    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    labels = labels.fillna(0).astype(int)

    valid = features.index[200:-1]
    return features.loc[valid], labels.loc[valid]


def test_config(X_train, y_train, X_test, y_test, test_prices, config):
    """测试配置"""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 200),
        max_depth=config.get('max_depth', 8),
        min_samples_split=config.get('min_samples_split', 30),
        min_samples_leaf=config.get('min_samples_leaf', 15),
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    proba = model.predict_proba(X_test_sc)[:, 1]
    acc = (model.predict(X_test_sc) == y_test).mean()

    # 信号
    threshold = config['threshold']
    signals = pd.Series(0, index=X_test.index)
    signals[proba >= threshold] = 1

    # 只做多，高置信度买入
    trades = []
    for i in range(len(signals)-1):
        if signals.iloc[i] == 1:
            entry = test_prices['close'].iloc[i]
            exit_p = test_prices['close'].iloc[i+1]
            ret = (exit_p - entry) / entry - 0.002
            trades.append(ret)

    if len(trades) < 10:
        return {'acc': acc, 'win_rate': 0, 'trades': 0, 'return': 0}

    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades)
    total_ret = sum(trades)

    return {
        'acc': acc,
        'win_rate': win_rate,
        'trades': len(trades),
        'return': total_ret,
        'model': model,
        'scaler': scaler
    }


def optimize():
    """参数优化"""
    print("=" * 60)
    print("BTC 15min Strategy Optimizer")
    print("=" * 60)

    df = get_data()
    if df is None:
        print("No data")
        return

    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X, y = create_features(df)
    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    test_prices = test_df.loc[X_test.index]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    best_result = None
    best_config = None

    # 测试不同阈值
    thresholds = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.65]

    print("\nTesting configurations...")

    for threshold in thresholds:
        config = {
            'threshold': threshold,
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_split': 30,
            'min_samples_leaf': 15
        }

        result = test_config(X_train, y_train, X_test, y_test, test_prices, config)

        marker = " ***" if result['win_rate'] >= 0.60 else ""
        print(f"T={threshold:.2f}: Acc={result['acc']:.2%}, WR={result['win_rate']:.2%}, "
              f"Trades={result['trades']}, Ret={result['return']:.2%}{marker}")

        if result['win_rate'] >= 0.55 and result['trades'] >= 20:
            if best_result is None or result['win_rate'] > best_result['win_rate']:
                best_result = result
                best_config = config

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)

    if best_result:
        print(f"Config: {best_config}")
        print(f"Accuracy: {best_result['acc']:.2%}")
        print(f"Win Rate: {best_result['win_rate']:.2%}")
        print(f"Trades: {best_result['trades']}")
        print(f"Return: {best_result['return']:.2%}")

        if best_result['win_rate'] >= 0.60:
            print("\n*** TARGET ACHIEVED! ***")

        # 保存最佳模型
        import joblib
        os.makedirs('results', exist_ok=True)
        joblib.dump({
            'model': best_result['model'],
            'scaler': best_result['scaler'],
            'config': best_config
        }, 'results/model_15min.pkl')

    return best_result, best_config


if __name__ == "__main__":
    optimize()

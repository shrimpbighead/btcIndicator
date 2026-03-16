"""
BTC 15分钟级别策略 - 最终版
基于遗传算法优化和ML预测的最佳策略
"""
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from config import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT


# 最佳参数 (从优化结果得出)
BEST_PARAMS = {
    'look_ahead': 4,
    'threshold': 0.6,
    'model_type': 'rf',
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_split': 30,
    'min_samples_leaf': 15
}


def load_or_fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=400):
    """加载数据"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"

    df = load_data(data_file)
    if df is not None:
        return df

    print("Fetching data...")
    try:
        fetcher = BinanceDataFetcher(symbol, timeframe, proxy=DEFAULT_PROXY)
        df = fetcher.fetch_historical_data(days=days)
        if df is not None:
            save_data(df, data_file)
    except Exception as e:
        print(f"Error: {e}")
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


def train_and_test():
    """训练和测试"""
    print("=" * 60)
    print("BTC 15m ML Strategy - Production Version")
    print("=" * 60)

    df = load_or_fetch_data()
    if df is None:
        print("No data")
        return

    # 分割
    split = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"\nData:")
    print(f"  Training: {len(train_df)} ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"  Testing: {len(test_df)} ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    # 创建特征
    params = BEST_PARAMS
    X, y = create_features(df, look_ahead=params['look_ahead'])
    split_idx = len(train_df) - 200

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"\nSamples:")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 训练
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    # 预测
    train_pred = model.predict(X_train_sc)
    test_pred = model.predict(X_test_sc)
    proba = model.predict_proba(X_test_sc)[:, 1]

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    print(f"\nModel Accuracy:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Test: {test_acc:.2%}")

    # 信号
    threshold = params['threshold']
    signals = pd.Series(0, index=X_test.index)
    signals[proba >= threshold] = 1
    signals[proba <= (1-threshold)] = -1

    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    print(f"\nSignals:")
    print(f"  Buy: {buy_signals}")
    print(f"  Sell: {sell_signals}")
    print(f"  Hold: {(signals == 0).sum()}")

    # 回测
    test_prices = test_df.loc[signals.index]

    trades = []
    pos = 0
    entry = 0
    entry_time = None
    trade_log = []

    for i in range(len(signals)-1):
        sig = signals.iloc[i]
        time = signals.index[i]
        price = test_prices['close'].iloc[i]

        if sig != 0 and pos == 0:
            pos = sig
            entry = price
            entry_time = time
        elif sig != 0 and pos != 0 and sig != pos:
            exit_p = price
            exit_time = time

            if pos == 1:
                ret = (exit_p - entry) / entry
                direction = 'LONG'
            else:
                ret = (entry - exit_p) / entry
                direction = 'SHORT'

            ret -= 0.002  # 手续费
            trades.append(ret)
            trade_log.append({
                'entry_time': str(entry_time),
                'exit_time': str(exit_time),
                'direction': direction,
                'entry': entry,
                'exit': exit_p,
                'return': ret
            })

            pos = sig
            entry = price
            entry_time = time

    # 最后平仓
    if pos != 0:
        exit_p = test_prices['close'].iloc[-1]
        exit_time = test_prices.index[-1]

        if pos == 1:
            ret = (exit_p - entry) / entry
            direction = 'LONG'
        else:
            ret = (entry - exit_p) / entry
            direction = 'SHORT'

        ret -= 0.002
        trades.append(ret)
        trade_log.append({
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'direction': direction,
            'entry': entry,
            'exit': exit_p,
            'return': ret
        })

    # 结果
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if len(trades) > 0:
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades)
        total_return = sum(trades)

        print(f"\nTotal Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_return:.2%}")

        # 最大连胜/连败
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in trades:
            if t > 0:
                streak = streak + 1 if streak >= 0 else 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = streak - 1 if streak <= 0 else -1
                max_loss_streak = max(max_loss_streak, abs(streak))

        print(f"Max Win Streak: {max_win_streak}")
        print(f"Max Loss Streak: {max_loss_streak}")

        if win_rate >= 0.60:
            print("\n*** TARGET ACHIEVED: Win Rate > 60% ***")

        # 保存交易记录
        os.makedirs('results', exist_ok=True)
        with open('results/trades.json', 'w') as f:
            json.dump(trade_log, f, indent=2)

        # 保存模型
        import joblib
        joblib.dump({'model': model, 'scaler': scaler, 'params': params}, 'results/model.pkl')

        print("\nResults saved to results/")
    else:
        print("No trades executed")

    # 返回结果
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'win_rate': win_rate if trades else 0,
        'total_return': sum(trades) if trades else 0,
        'total_trades': len(trades),
        'params': params
    }


if __name__ == "__main__":
    results = train_and_test()

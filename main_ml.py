"""
BTC 15分钟级别策略 - ML增强版
使用遗传算法优化特征选择 + 机器学习预测方向
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

from data.fetcher import BinanceDataFetcher, load_data, save_data, DEFAULT_PROXY
from indicators.calculator import calculate_all_indicators, get_feature_vector
from genetic.chromosome import Chromosome
from genetic.population import Population
from genetic.fitness import evaluate_population
from filter.signal_filter import SignalFilter
from backtest.engine import BacktestEngine
from config import SYMBOL, TIMEFRAME, TRAIN_TEST_SPLIT, USE_AI_FILTER, ML_MODEL


# 配置
USE_MOCK_DATA = False
NUM_GENERATIONS = 20
POPULATION_SIZE = 30


def load_or_fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=400):
    """加载或获取数据"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"

    df = load_data(data_file)
    if df is not None:
        print(f"Loaded existing data: {len(df)} candles")
        return df

    if not USE_MOCK_DATA:
        print(f"Fetching data from Binance...")
        try:
            fetcher = BinanceDataFetcher(symbol, timeframe, proxy=DEFAULT_PROXY)
            df = fetcher.fetch_historical_data(days=days)
            if df is not None:
                save_data(df, data_file)
                return df
        except Exception as e:
            print(f"Failed: {e}")

    return None


def create_features_and_labels(df, look_ahead=4):
    """
    创建机器学习特征和标签
    标签: 未来look_ahead根K线的价格变化方向
    """
    print("Creating features and labels...")

    # 创建基础特征
    features = pd.DataFrame(index=df.index)

    # 1. 价格变化率
    for period in [1, 2, 3, 4, 5, 8, 12, 16, 24]:
        features[f'return_{period}'] = df['close'].pct_change(period)

    # 2. 移动平均线
    for period in [5, 10, 20, 40, 60]:
        ma = df['close'].rolling(period).mean()
        features[f'price_to_ma_{period}'] = (df['close'] - ma) / ma

    # 3. 波动率
    for period in [8, 16, 24]:
        features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()

    # 4. RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # 5. MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = macd - signal

    # 6. 成交量特征
    features['volume_change'] = df['volume'].pct_change()
    features['volume_ma5'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()

    # 7. 价格位置
    features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # 8. 创建标签: 未来价格方向
    future_return = df['close'].shift(-look_ahead) / df['close'] - 1
    labels = (future_return > 0).astype(int)  # 1=上涨, 0=下跌

    # 填充NaN
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    # 移除没有足够数据的行
    valid_idx = features.index[200:-look_ahead]  # 跳过预热期和未来数据
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]

    print(f"Created {len(features)} samples")
    print(f"Class distribution: {labels.value_counts().to_dict()}")

    return features, labels


def train_ml_model(X_train, y_train, model_type='random_forest'):
    """训练ML模型"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    model.fit(X_train, y_train)
    return model


def run_ml_strategy(train_df, test_df, look_ahead=4):
    """运行ML策略"""
    print("\n" + "=" * 60)
    print("ML-Based Strategy")
    print("=" * 60)

    # 合并数据以创建特征
    combined_df = pd.concat([train_df, test_df])

    # 创建特征和标签
    features, labels = create_features_and_labels(combined_df, look_ahead)

    # 分割训练和测试
    split_idx = len(train_df) - 200  # 减去预热期
    train_end = combined_df.index.get_loc(train_df.index[-1])

    # 找到对应的特征索引
    feature_split_idx = 0
    for i, idx in enumerate(features.index):
        if idx >= train_df.index[-1]:
            feature_split_idx = i
            break

    X_train = features.iloc[:feature_split_idx]
    y_train = labels.iloc[:feature_split_idx]
    X_test = features.iloc[feature_split_idx:]
    y_test = labels.iloc[feature_split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # 训练模型
    print("\nTraining ML model...")
    model = train_ml_model(X_train, y_train, 'random_forest')

    # 训练集评估
    train_pred = model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(f"Training accuracy: {train_accuracy:.2%}")

    # 测试集预测
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = (test_pred == y_test).mean()
    print(f"Test accuracy: {test_accuracy:.2%}")

    # 生成交易信号
    # 置信度阈值
    threshold = 0.55  # 预测概率 > 0.55 时下单
    signals = pd.Series(0, index=X_test.index)
    signals[test_proba > threshold] = 1   # 买入
    signals[test_proba < (1-threshold)] = -1  # 卖出

    print(f"\nSignals generated: {(signals != 0).sum()}")
    print(f"Buy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")

    # 回测
    # 使用测试集的原始索引
    test_prices = test_df.copy()

    # 找到测试集在features中的位置
    test_start = test_df.index[0]
    test_end = test_df.index[-1]

    # 筛选测试集对应的特征
    test_mask = (features.index >= test_start) & (features.index <= test_end)
    test_features = features[test_mask]
    test_signals = signals.loc[test_features.index]

    print(f"\nSignals for backtest: {len(test_signals)}")

    # 手动计算回测
    trades = []
    position = 0
    entry_price = 0

    for i in range(len(test_signals)):
        signal = test_signals.iloc[i]
        # 获取对应的价格
        signal_time = test_signals.index[i]
        if signal_time in test_prices.index:
            price = test_prices.loc[signal_time, 'close']
        else:
            continue

        if signal != 0 and position == 0:
            # 开仓
            position = signal
            entry_price = price
        elif signal == 0 and position != 0:
            # 平仓
            exit_price = price
            if position == 1:  # 多头
                profit = (exit_price - entry_price) / entry_price
            else:  # 空头
                profit = (entry_price - exit_price) / entry_price
            profit -= 0.002  # 手续费
            trades.append({
                'profit': profit,
                'type': 'long' if position == 1 else 'short'
            })
            position = 0

    # 最后如果还有仓位
    if position != 0:
        exit_price = test_prices['close'].iloc[-1]
        if position == 1:
            profit = (exit_price - entry_price) / entry_price
        else:
            profit = (entry_price - exit_price) / entry_price
        profit -= 0.002
        trades.append({'profit': profit, 'type': 'long' if position == 1 else 'short'})

    # 计算结果
    if trades:
        trades_df = pd.DataFrame(trades)
        wins = trades_df['profit'] > 0
        win_rate = wins.sum() / len(trades_df)
        total_return = trades_df['profit'].sum()

        print(f"\n=== Backtest Results ===")
        print(f"Total trades: {len(trades_df)}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Total return: {total_return:.2%}")

        return {
            'win_rate': win_rate,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'test_accuracy': test_accuracy,
            'model': model,
            'features': features
        }

    return {
        'win_rate': 0,
        'total_return': 0,
        'total_trades': 0,
        'test_accuracy': test_accuracy,
        'model': model,
        'features': features
    }


def main():
    """主函数"""
    print("=" * 60)
    print("BTC 15m ML Strategy Optimizer")
    print("Target: Win Rate > 60%")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] Loading data...")
    df = load_or_fetch_data()
    if df is None:
        print("Failed to load data")
        return

    print(f"Total: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    # 2. 分割数据
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Training: {len(train_df)} ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Testing: {len(test_df)} ({test_df.index[0]} to {test_df.index[-1]})")

    # 3. 运行ML策略
    results = run_ml_strategy(train_df, test_df)

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")

    if results['win_rate'] >= 0.60:
        print("\n*** TARGET ACHIEVED: Win Rate > 60% ***")
    else:
        print(f"\n*** TARGET NOT MET: Win Rate is {results['win_rate']:.2%} ***")


if __name__ == "__main__":
    main()

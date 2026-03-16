"""
BTC 15分钟级别策略优化系统 - 主程序
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

# 导入项目模块
from data.fetcher import BinanceDataFetcher, load_data, save_data, generate_mock_btc_data, DEFAULT_PROXY
from indicators.calculator import calculate_all_indicators, generate_signals
from genetic.algorithm import GeneticAlgorithm
from genetic.fitness import calculate_fitness
from filter.signal_filter import SignalFilter
from backtest.engine import BacktestEngine
from config import SYMBOL, TIMEFRAME, DATA_LIMIT, TRAIN_TEST_SPLIT, USE_AI_FILTER, ML_MODEL


# 是否使用模拟数据 (当网络不可用时)
USE_MOCK_DATA = False


def load_or_fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=400):
    """加载或获取数据"""
    # 数据文件路径
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"

    # 尝试加载已有数据
    df = load_data(data_file)

    if df is not None:
        print(f"Loaded existing data: {len(df)} candles")
        return df

    # 尝试获取真实数据
    if not USE_MOCK_DATA:
        print(f"Fetching data from Binance (proxy: {DEFAULT_PROXY})...")
        try:
            fetcher = BinanceDataFetcher(symbol, timeframe, proxy=DEFAULT_PROXY)
            df = fetcher.fetch_historical_data(days=days)

            if df is not None and len(df) > 0:
                print(f"Fetched {len(df)} candles")
                save_data(df, data_file)
                return df
        except Exception as e:
            print(f"Failed to fetch from Binance: {e}")
            print("Falling back to mock data...")

    # 使用模拟数据
    print("Generating mock BTC data for testing...")
    df = generate_mock_btc_data(n_candles=3000, base_price=45000)
    save_data(df, data_file)
    return df


def split_data(df, train_ratio=TRAIN_TEST_SPLIT):
    """分割训练集和测试集"""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nData split:")
    print(f"  Training: {len(train_df)} candles ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Testing: {len(test_df)} candles ({test_df.index[0]} to {test_df.index[-1]})")

    return train_df, test_df


def run_optimization(train_df):
    """运行遗传算法优化"""
    print("\n" + "=" * 60)
    print("PHASE 1: Genetic Algorithm Optimization")
    print("=" * 60)

    ga = GeneticAlgorithm(train_df)
    best_solution, history = ga.run()

    return best_solution, history


def train_ai_filter(train_df, params):
    """训练AI信号过滤器"""
    if not USE_AI_FILTER:
        return None

    print("\n" + "=" * 60)
    print("PHASE 2: AI Signal Filter Training")
    print("=" * 60)

    # 计算指标
    df_with_indicators = calculate_all_indicators(train_df.copy(), params)

    # 生成信号
    signals = generate_signals(df_with_indicators, params)

    # 训练过滤器
    signal_filter = SignalFilter(model_type=ML_MODEL)
    success = signal_filter.train(train_df, params, signals)

    if success:
        # 保存模型
        signal_filter.save('models/signal_filter.pkl')
        return signal_filter
    else:
        return None


def run_backtest(test_df, params, signal_filter=None):
    """运行回测"""
    print("\n" + "=" * 60)
    print("PHASE 3: Backtesting")
    print("=" * 60)

    # 计算指标
    df_with_indicators = calculate_all_indicators(test_df.copy(), params)

    # 生成信号
    signals = generate_signals(df_with_indicators, params)

    # 应用AI过滤器
    if signal_filter is not None:
        print("Applying AI signal filter...")
        signals, confidence = signal_filter.filter_signals(df_with_indicators, signals)
        print(f"Original signals: {(signals != 0).sum()}")
        print(f"Filtered signals: {(signals != 0).sum()}")

    # 运行回测
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(test_df, signals)

    print("\nBacktest Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Final Capital: ${results['final_capital']:.2f}")

    return results, signals


def plot_results(history, backtest_results, params):
    """绘制结果图表"""
    # 创建图表目录
    os.makedirs('results', exist_ok=True)

    # 1. GA优化历史
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 适应度曲线
    ax1 = axes[0, 0]
    generations = [h['generation'] for h in history]
    fitness = [h['best_fitness'] for h in history]
    ax1.plot(generations, fitness, 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Genetic Algorithm - Best Fitness')
    ax1.grid(True)

    # 胜率曲线
    ax2 = axes[0, 1]
    win_rates = [h['best_win_rate'] for h in history]
    ax2.plot(generations, win_rates, 'g-', linewidth=2)
    ax2.axhline(y=0.6, color='r', linestyle='--', label='Target 60%')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate Progress')
    ax2.legend()
    ax2.grid(True)

    # 权益曲线
    ax3 = axes[1, 0]
    if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
        trades = backtest_results['trades']
        equity = [10000]
        for _, trade in trades.iterrows():
            equity.append(equity[-1] * (1 + trade['return']))
        ax3.plot(equity, 'b-', linewidth=2)
    ax3.set_xlabel('Trade')
    ax3.set_ylabel('Equity ($)')
    ax3.set_title('Equity Curve')
    ax3.grid(True)

    # 收益分布
    ax4 = axes[1, 1]
    if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
        returns = backtest_results['trades']['return'] * 100
        ax4.hist(returns, bins=30, edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Return Distribution')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('results/optimization_results.png', dpi=150)
    print("\nResults saved to results/optimization_results.png")

    plt.close()


def save_results(best_solution, history, backtest_results):
    """保存结果"""
    os.makedirs('results', exist_ok=True)

    # 保存最佳参数
    with open('results/best_params.json', 'w') as f:
        json.dump(best_solution.to_dict(), f, indent=2)

    # 保存历史
    with open('results/optimization_history.json', 'w') as f:
        # 转换numpy类型
        for h in history:
            h['best_genes'] = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                             for k, v in h['best_genes'].items()}
        json.dump(history, f, indent=2)

    # 保存回测结果
    results_to_save = {k: v for k, v in backtest_results.items() if k != 'trades'}
    with open('results/backtest_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print("Results saved to results/")


def main():
    """主函数"""
    print("=" * 60)
    print("BTC 15m Strategy Optimizer")
    print("Target: Win Rate > 60%")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] Loading data...")
    df = load_or_fetch_data()

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Total data: {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # 2. 分割数据
    train_df, test_df = split_data(df)

    # 3. 遗传算法优化
    best_solution, history = run_optimization(train_df)

    if best_solution is None:
        print("Optimization failed. Exiting.")
        return

    # 4. 训练AI过滤器
    signal_filter = None
    if USE_AI_FILTER:
        signal_filter = train_ai_filter(train_df, best_solution.to_dict())

    # 5. 回测
    backtest_results, final_signals = run_backtest(test_df, best_solution.to_dict(), signal_filter)

    # 6. 保存和可视化
    print("\n[4/4] Saving results...")
    plot_results(history, backtest_results, best_solution.to_dict())
    save_results(best_solution, history, backtest_results)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Parameters: {best_solution.to_dict()}")
    print(f"\nTest Set Performance:")
    print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"  Total Return: {backtest_results['total_return']:.2%}")

    if backtest_results['win_rate'] >= 0.60:
        print("\n*** TARGET ACHIEVED: Win Rate > 60% ***")
    else:
        print(f"\n*** TARGET NOT MET: Win Rate is {backtest_results['win_rate']:.2%} (target: 60%) ***")


if __name__ == "__main__":
    main()

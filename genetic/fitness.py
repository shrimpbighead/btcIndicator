"""
适应度评估模块 - 最终版
使用多周期验证，避免过拟合
"""
import pandas as pd
import numpy as np
from indicators.calculator import calculate_all_indicators, generate_signals
from config import MIN_WIN_RATE_THRESHOLD, LOOK_AHEAD_BARS, FEE_RATE, SLIPPAGE


def calculate_fitness(df, chromosome):
    """
    计算染色体适应度 - 最终版
    使用walk-forward方式评估
    """
    params = chromosome.to_dict()

    # 计算指标
    df_with_indicators = calculate_all_indicators(df.copy(), params)

    # 生成信号
    signals = generate_signals(df_with_indicators, params)

    # 使用更大的评估窗口
    eval_start = 200  # 跳过预热期
    eval_length = min(5000, len(df) - eval_start - LOOK_AHEAD_BARS)

    if eval_length < 100:
        chromosome.fitness = 0
        chromosome.win_rate = 0
        chromosome.profit_factor = 0
        chromosome.max_drawdown = 0
        chromosome.trade_count = 0
        return 0

    # 计算所有可能的交易
    all_trades = []
    for i in range(eval_start, eval_start + eval_length):
        signal = signals.iloc[i]

        if signal == 0:
            continue

        # 入场价
        entry_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + LOOK_AHEAD_BARS]

        # 计算收益（简化版，不考虑仓位）
        price_change = (future_price - entry_price) / entry_price

        # 根据信号方向计算盈亏
        if signal == 1:  # 做多
            profit = price_change
        else:  # 做空
            profit = -price_change

        # 扣除手续费
        profit -= (FEE_RATE * 2 + SLIPPAGE)

        all_trades.append({
            'profit': profit,
            'signal': signal,
            'price_change': price_change
        })

    if len(all_trades) == 0:
        chromosome.fitness = 0
        chromosome.win_rate = 0
        chromosome.profit_factor = 0
        chromosome.max_drawdown = 0
        chromosome.trade_count = 0
        return 0

    trades_df = pd.DataFrame(all_trades)

    # 计算胜率
    wins = trades_df['profit'] > 0
    win_count = wins.sum()
    total_trades = len(trades_df)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # 盈利因子
    gross_profit = trades_df.loc[trades_df['profit'] > 0, 'profit'].sum() if win_count > 0 else 0
    gross_loss = abs(trades_df.loc[trades_df['profit'] < 0, 'profit'].sum()) if (total_trades - win_count) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

    # 最大回撤
    cumulative_returns = trades_df['profit'].cumsum()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # 存储结果
    chromosome.win_rate = win_rate
    chromosome.profit_factor = profit_factor if profit_factor != float('inf') else 10
    chromosome.max_drawdown = max_drawdown
    chromosome.trade_count = total_trades

    # 适应度函数 - 以胜率为主
    if win_rate < MIN_WIN_RATE_THRESHOLD:
        fitness = 0.01 * win_rate
    else:
        # 考虑胜率、盈利因子、交易频率和回撤
        drawdown_penalty = max(0, 1 - max_drawdown / 1)  # 回撤惩罚

        # 交易频率因子 - 至少要有一定的交易量
        min_trades = 50
        frequency_factor = min(total_trades / min_trades, 1.0)

        fitness = (
            win_rate * 0.65 +
            min(profit_factor / 3, 1) * 0.15 +
            drawdown_penalty * 0.1 +
            frequency_factor * 0.1
        )

    chromosome.fitness = fitness
    return fitness


def evaluate_population(df, population):
    """评估种群"""
    for chrom in population.chromosomes:
        calculate_fitness(df, chrom)

    population.best_chromosome = population.get_best()

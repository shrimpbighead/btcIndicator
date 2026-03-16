"""
BTC 15分钟策略 - 正确逻辑版
每根K线收盘时做决策，下一根K线收盘时平仓
"""
import pandas as pd
import numpy as np
import json

from data.fetcher import load_data


def add_indicators(df):
    """计算指标"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    return df


def run_strategy():
    """运行策略"""
    df = load_data("data/BTC_USDT_15m.csv")

    print("=" * 60)
    print("BTC 15分钟策略 - 每根K线交易")
    print("=" * 60)
    print(f"数据: {len(df)} 根K线")
    print(f"时间: {df.index[0]} 到 {df.index[-1]}")

    # 最佳参数
    params = {'rsi_buy': 35, 'rsi_sell': 80, 'threshold': 2}

    # 计算指标
    df = add_indicators(df.copy())

    # 生成信号
    scores = pd.Series(0.0, index=df.index)

    # RSI信号
    scores[df['rsi'] < params['rsi_buy']] += 1      # RSI超卖 -> 买入
    scores[df['rsi'] > params['rsi_sell']] -= 1      # RSI超买 -> 卖出

    # MACD金叉死叉
    macd_buy = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
    macd_sell = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
    scores[macd_buy] += 1
    scores[macd_sell] -= 1

    # 信号
    signals = pd.Series(0, index=df.index)
    signals[scores >= params['threshold']] = 1
    signals[scores <= -params['threshold']] = -1

    # 模拟交易 - 从第200根开始（指标预热）
    trades = []
    for i in range(200, len(df) - 1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        # 入场：当前K线收盘价
        entry_time = df.index[i]
        entry_price = df.iloc[i]['close']

        # 平仓：下一根K线收盘价
        exit_time = df.index[i + 1]
        exit_price = df.iloc[i + 1]['close']

        # 判断盈亏
        if sig == 1:  # 做多
            is_win = exit_price > entry_price
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:  # 做空
            is_win = exit_price < entry_price
            return_pct = (entry_price - exit_price) / entry_price * 100

        trades.append({
            'id': len(trades) + 1,
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'direction': 'LONG' if sig == 1 else 'SHORT',
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'return_pct': round(float(return_pct), 2),
            'success': bool(is_win),
            'rsi_entry': round(float(df.iloc[i]['rsi']), 2),
            'macd': round(float(df.iloc[i]['macd']), 2),
            'macd_signal': round(float(df.iloc[i]['macd_sig']), 2)
        })

    # 统计
    total = len(trades)
    wins = sum(1 for t in trades if t['success'])
    losses = total - wins
    win_rate = wins / total * 100 if total > 0 else 0
    total_return = sum(t['return_pct'] for t in trades)

    print(f"\n总交易: {total}")
    print(f"成功: {wins}")
    print(f"失败: {losses}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"总收益: {total_return:.2f}%")

    # 保存
    with open('results/trades_15min_full.json', 'w') as f:
        json.dump(trades, f, indent=2)

    print(f"\n已保存到: results/trades_15min_full.json")

    # 显示前20笔
    print("\n最近20笔交易:")
    for t in trades[-20:]:
        s = "成功" if t['success'] else "失败"
        print(f"{t['id']:3d}. {t['entry_time'][:16]} | {t['direction']:5s} | "
              f"{t['entry_price']:.0f} -> {t['exit_price']:.0f} | {t['return_pct']:+6.2f}% | {s}")


if __name__ == "__main__":
    run_strategy()

"""
BTC 15分钟策略 - 详细分析版
"""
import pandas as pd
import numpy as np
import json

from data.fetcher import load_data


def add_indicators(df):
    # MA
    for p in [5, 10, 20]:
        df[f'ma{p}'] = df['close'].rolling(p).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_sig'] = df['macd'].ewm(span=9).mean()

    # Stochastic
    low = df['low'].rolling(14).min()
    high = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low) / (high - low + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    return df


def run_strategy_detailed():
    # 加载数据
    df = load_data("data/BTC_USDT_15m.csv")

    print("=" * 70)
    print("BTC 15分钟策略 - 详细分析")
    print("=" * 70)

    print(f"\n总数据量: {len(df)} 根K线")
    print(f"数据时间: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据跨度: {(df.index[-1] - df.index[0]).days} 天")

    # 分割数据
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"\n训练集: {len(train_df)} 根K线")
    print(f"  时间: {train_df.index[0]} 到 {train_df.index[-1]}")
    print(f"\n测试集: {len(test_df)} 根K线")
    print(f"  时间: {test_df.index[0]} 到 {test_df.index[-1]}")

    # 最佳参数
    params = {
        'rsi_buy': 35,
        'rsi_sell': 80,
        'macd': 1,
        'stoch': 0,
        'bb': 0,
        'threshold': 2
    }

    # 在测试集上运行
    test_df = add_indicators(test_df.copy())

    scores = pd.Series(0.0, index=test_df.index)

    # RSI
    scores[test_df['rsi'] < params['rsi_buy']] += 1
    scores[test_df['rsi'] > params['rsi_sell']] -= 1

    # MACD
    if params['macd']:
        buy = (test_df['macd'] > test_df['macd_sig']) & (test_df['macd'].shift(1) <= test_df['macd_sig'].shift(1))
        sell = (test_df['macd'] < test_df['macd_sig']) & (test_df['macd'].shift(1) >= test_df['macd_sig'].shift(1))
        scores[buy] += 1
        scores[sell] -= 1

    # 信号
    signals = pd.Series(0, index=test_df.index)
    signals[scores >= params['threshold']] = 1
    signals[scores <= -params['threshold']] = -1

    # 回测并记录每笔交易
    trades = []
    for i in range(1, len(test_df)-1):
        sig = signals.iloc[i]
        if sig == 0:
            continue

        entry_time = test_df.index[i]
        exit_time = test_df.index[i+1]
        entry_price = test_df.iloc[i]['close']
        exit_price = test_df.iloc[i+1]['close']

        if sig == 1:  # 做多
            direction = "LONG"
            return_pct = (exit_price - entry_price) / entry_price * 100
            success = exit_price > entry_price
        else:  # 做空
            direction = "SHORT"
            return_pct = (entry_price - exit_price) / entry_price * 100
            success = exit_price < entry_price

        trades.append({
            'id': len(trades) + 1,
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(return_pct, 2),
            'success': bool(success),  # 转换为Python bool
            'rsi_entry': round(test_df.iloc[i]['rsi'], 2),
            'macd_entry': round(test_df.iloc[i]['macd'], 2),
            'macd_sig_entry': round(test_df.iloc[i]['macd_sig'], 2)
        })

    # 统计
    print("\n" + "=" * 70)
    print("测试集交易统计")
    print("=" * 70)

    total_trades = len(trades)
    successful_trades = sum(1 for t in trades if t['success'])
    failed_trades = total_trades - successful_trades
    win_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0

    print(f"\n总交易次数: {total_trades}")
    print(f"成功次数: {successful_trades}")
    print(f"失败次数: {failed_trades}")
    print(f"胜率: {win_rate:.2f}%")

    # 按方向统计
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']

    print(f"\n做多交易: {len(longs)} 次")
    print(f"做空交易: {len(shorts)} 次")

    if longs:
        long_wins = sum(1 for t in longs if t['success'])
        print(f"  胜率: {long_wins/len(longs)*100:.2f}%")

    if shorts:
        short_wins = sum(1 for t in shorts if t['success'])
        print(f"  胜率: {short_wins/len(shorts)*100:.2f}%")

    # 收益统计
    total_return = sum(t['return_pct'] for t in trades)
    avg_return = total_return / total_trades if total_trades > 0 else 0

    print(f"\n总收益: {total_return:.2f}%")
    print(f"平均每笔收益: {avg_return:.2f}%")

    # 最大连胜/连败
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for t in trades:
        if t['success']:
            streak = streak + 1 if streak >= 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak <= 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    print(f"\n最大连胜: {max_win_streak} 次")
    print(f"最大连败: {max_loss_streak} 次")

    # 保存交易详情
    with open('results/trades_detail.json', 'w') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)

    print(f"\n交易详情已保存到: results/trades_detail.json")

    # 打印前10笔交易
    print("\n" + "=" * 70)
    print("最近10笔交易")
    print("=" * 70)
    for t in trades[-10:]:
        status = "✓" if t['success'] else "✗"
        print(f"{t['id']:2d}. {t['entry_time'][:19]} | {t['direction']:5s} | "
              f"入:{t['entry_price']:.2f} 出:{t['exit_price']:.2f} | "
              f"收益:{t['return_pct']:+6.2f}% | {status}")

    return trades


if __name__ == "__main__":
    trades = run_strategy_detailed()

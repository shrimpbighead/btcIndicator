"""
回测引擎
"""
import pandas as pd
import numpy as np
from config import FEE_RATE, SLIPPAGE, LOOK_AHEAD_BARS


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 持仓数量
        self.position_type = None  # 'long' or 'short'
        self.trades = []
        self.equity_curve = []

    def reset(self):
        """重置状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.position_type = None
        self.trades = []
        self.equity_curve = []

    def execute_trade(self, entry_time, entry_price, signal, size=1.0):
        """
        执行交易

        Args:
            entry_time: 入场时间
            entry_price: 入场价格
            signal: 1=买入(做多), -1=卖出(做空), 0=平仓
            size: 仓位大小 (0-1, 占总资金比例)
        """
        if signal == 0:
            # 平仓
            if self.position != 0:
                exit_price = entry_price
                profit = self.calculate_profit(entry_price, exit_price, self.position_type)
                self.capital += profit

                self.trades.append({
                    'entry_time': self.trades[-1]['entry_time'] if self.trades else entry_time,
                    'exit_time': entry_time,
                    'type': self.position_type,
                    'entry_price': self.trades[-1]['entry_price'] if self.trades else entry_price,
                    'exit_price': exit_price,
                    'size': abs(self.position),
                    'profit': profit,
                    'return': profit / (self.trades[-1]['entry_price'] * abs(self.position)) if self.trades else 0
                })

                self.position = 0
                self.position_type = None
            return

        # 开仓
        if self.position == 0:
            # 计算买入数量
            trade_capital = self.capital * size
            # 考虑手续费
            effective_capital = trade_capital / (1 + FEE_RATE + SLIPPAGE)
            self.position = effective_capital / entry_price
            self.position_type = 'long' if signal == 1 else 'short'

            self.trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'type': self.position_type,
                'size': self.position
            })

    def calculate_profit(self, entry_price, exit_price, position_type):
        """计算收益"""
        if position_type == 'long':
            # 做多：(exit_price - entry_price) * quantity - fees
            profit = (exit_price - entry_price) * self.position
        else:
            # 做空：(entry_price - exit_price) * quantity - fees
            profit = (entry_price - exit_price) * self.position

        # 扣除手续费
        fees = (entry_price + exit_price) * self.position * (FEE_RATE + SLIPPAGE)
        return profit - fees

    def run_backtest(self, df, signals):
        """
        运行回测

        Args:
            df: 价格数据
            signals: 交易信号

        Returns:
            回测结果
        """
        self.reset()

        for i in range(len(df) - 1):
            signal = signals.iloc[i]
            price = df['close'].iloc[i]
            time = df.index[i]

            # 执行交易
            self.execute_trade(time, price, signal)

            # 记录权益
            position_value = self.position * price if self.position > 0 else 0
            equity = self.capital + position_value
            self.equity_curve.append({
                'time': time,
                'equity': equity,
                'capital': self.capital,
                'position': self.position
            })

        # 平掉所有仓位
        if self.position > 0:
            last_price = df['close'].iloc[-1]
            last_time = df.index[-1]
            profit = self.calculate_profit(self.trades[-1]['entry_price'], last_price, self.position_type)
            self.capital += profit
            self.trades[-1]['exit_time'] = last_time
            self.trades[-1]['exit_price'] = last_price
            self.trades[-1]['profit'] = profit
            self.trades[-1]['return'] = profit / (self.trades[-1]['entry_price'] * self.position)
            self.position = 0

        # 计算指标
        return self.calculate_metrics()

    def calculate_metrics(self):
        """计算回测指标"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # 过滤已平仓的交易
        closed_trades = trades_df[trades_df['profit'].notna()]
        if len(closed_trades) == 0:
            return {
                'total_trades': len(trades_df),
                'closed_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': (self.capital - self.initial_capital) / self.initial_capital,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade': 0,
                'final_capital': self.capital
            }

        wins = closed_trades[closed_trades['profit'] > 0]
        losses = closed_trades[closed_trades['profit'] <= 0]

        win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0

        gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # 计算权益曲线
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()
            max_drawdown = abs(equity_df['drawdown'].min())
        else:
            max_drawdown = 0

        # 夏普比率
        if len(closed_trades) > 1:
            returns = closed_trades['return']
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        total_return = (self.capital - self.initial_capital) / self.initial_capital

        return {
            'total_trades': len(trades_df),
            'closed_trades': len(closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 100,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade': closed_trades['profit'].mean() if len(closed_trades) > 0 else 0,
            'trades': closed_trades,
            'final_capital': self.capital
        }

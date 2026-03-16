"""
遗传算法染色体编码
"""
import random
import numpy as np
from config import *


class Chromosome:
    """染色体类 - 代表一组指标参数"""

    def __init__(self, genes=None):
        if genes is None:
            self.genes = self._random_genes()
        else:
            self.genes = genes

        self.fitness = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.max_drawdown = 0
        self.trade_count = 0

    def _random_genes(self):
        """随机生成基因"""
        genes = {
            # MA参数 (选择1-2个)
            'ma_period': random.choice(MA_PERIODS),

            # EMA参数
            'ema_period': random.choice(EMA_PERIODS),

            # MACD参数
            'macd_fast': random.choice(MACD_FAST),
            'macd_slow': random.choice(MACD_SLOW),
            'macd_signal': random.choice(MACD_SIGNAL),

            # RSI参数
            'rsi_period': random.choice(RSI_PERIODS),

            # Stochastic参数
            'stoch_k': random.choice(STOCH_K),
            'stoch_d': random.choice(STOCH_D),

            # CCI参数
            'cci_period': random.choice(CCI_PERIODS),

            # Bollinger Bands参数
            'bb_period': random.choice(BB_PERIODS),
            'bb_std': random.choice(BB_STD),

            # ATR参数
            'atr_period': random.choice(ATR_PERIODS),
        }
        return genes

    def to_dict(self):
        """转换为字典"""
        return self.genes.copy()

    def copy(self):
        """复制染色体"""
        new_chrom = Chromosome(self.genes.copy())
        new_chrom.fitness = self.fitness
        new_chrom.win_rate = self.win_rate
        new_chrom.profit_factor = self.profit_factor
        new_chrom.max_drawdown = self.max_drawdown
        new_chrom.trade_count = self.trade_count
        return new_chrom

    def mutate(self, mutation_rate=MUTATION_RATE):
        """基因变异"""
        new_genes = self.genes.copy()

        for key in new_genes:
            if random.random() < mutation_rate:
                # 根据不同的基因进行不同的变异
                if key == 'ma_period':
                    new_genes[key] = random.choice(MA_PERIODS)
                elif key == 'ema_period':
                    new_genes[key] = random.choice(EMA_PERIODS)
                elif key == 'macd_fast':
                    new_genes[key] = random.choice(MACD_FAST)
                elif key == 'macd_slow':
                    new_genes[key] = random.choice(MACD_SLOW)
                elif key == 'macd_signal':
                    new_genes[key] = random.choice(MACD_SIGNAL)
                elif key == 'rsi_period':
                    new_genes[key] = random.choice(RSI_PERIODS)
                elif key == 'stoch_k':
                    new_genes[key] = random.choice(STOCH_K)
                elif key == 'stoch_d':
                    new_genes[key] = random.choice(STOCH_D)
                elif key == 'cci_period':
                    new_genes[key] = random.choice(CCI_PERIODS)
                elif key == 'bb_period':
                    new_genes[key] = random.choice(BB_PERIODS)
                elif key == 'bb_std':
                    new_genes[key] = random.choice(BB_STD)
                elif key == 'atr_period':
                    new_genes[key] = random.choice(ATR_PERIODS)

        return Chromosome(new_genes)

    @staticmethod
    def crossover(parent1, parent2, crossover_rate=CROSSOVER_RATE):
        """交叉"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_genes = {}
        child2_genes = {}

        # 单点交叉
        keys = list(parent1.genes.keys())
        split_point = random.randint(1, len(keys) - 1)

        for i, key in enumerate(keys):
            if i < split_point:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]

        return Chromosome(child1_genes), Chromosome(child2_genes)

    def __str__(self):
        return f"Chromosome(fitness={self.fitness:.4f}, win_rate={self.win_rate:.4f}, trades={self.trade_count})"

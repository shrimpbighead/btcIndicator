"""
种群管理
"""
import random
from genetic.chromosome import Chromosome
from config import POPULATION_SIZE, ELITE_COUNT


class Population:
    """种群管理类"""

    def __init__(self, size=POPULATION_SIZE):
        self.size = size
        self.chromosomes = []
        self.best_chromosome = None
        self.generation = 0

    def initialize(self):
        """初始化种群"""
        self.chromosomes = [Chromosome() for _ in range(self.size)]
        self.evaluate_fitness()
        self.best_chromosome = self.get_best()

    def evaluate_fitness(self):
        """评估所有染色体的适应度（由外部评估函数设置）"""
        pass

    def get_best(self):
        """获取最佳染色体"""
        if not self.chromosomes:
            return None
        return max(self.chromosomes, key=lambda x: x.fitness)

    def get_elite(self, count=ELITE_COUNT):
        """获取精英个体"""
        sorted_chroms = sorted(self.chromosomes, key=lambda x: x.fitness, reverse=True)
        return sorted_chroms[:count]

    def select_parent(self, tournament_size=3):
        """锦标赛选择"""
        tournament = random.sample(self.chromosomes, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve(self):
        """进化一代"""
        new_chromosomes = []

        # 保留精英个体
        elites = self.get_elite()
        new_chromosomes.extend([elite.copy() for elite in elites])

        # 生成新个体
        while len(new_chromosomes) < self.size:
            # 选择父母
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            # 交叉
            child1, child2 = Chromosome.crossover(parent1, parent2)

            # 变异
            child1 = child1.mutate()
            child2 = child2.mutate()

            new_chromosomes.append(child1)
            if len(new_chromosomes) < self.size:
                new_chromosomes.append(child2)

        self.chromosomes = new_chromosomes[:self.size]
        self.generation += 1

    def get_statistics(self):
        """获取种群统计信息"""
        fitness_values = [c.fitness for c in self.chromosomes]
        win_rates = [c.win_rate for c in self.chromosomes if c.trade_count > 0]

        return {
            'generation': self.generation,
            'best_fitness': self.best_chromosome.fitness if self.best_chromosome else 0,
            'best_win_rate': self.best_chromosome.win_rate if self.best_chromosome else 0,
            'avg_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else 0,
        }

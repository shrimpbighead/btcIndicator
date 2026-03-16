"""
遗传算法主模块
"""
from tqdm import tqdm
from genetic.population import Population
from genetic.fitness import evaluate_population
from config import NUM_GENERATIONS, POPULATION_SIZE


class GeneticAlgorithm:
    """遗传算法优化器"""

    def __init__(self, df):
        self.df = df
        self.population = Population(POPULATION_SIZE)
        self.best_solution = None
        self.history = []

    def run(self):
        """运行遗传算法"""
        print("Initializing population...")
        self.population.initialize()
        evaluate_population(self.df, self.population)

        print(f"\nStarting genetic algorithm optimization...")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Generations: {NUM_GENERATIONS}")

        for generation in tqdm(range(NUM_GENERATIONS), desc="Genetic Algorithm"):
            # 记录当前最佳
            current_best = self.population.get_best()
            self.history.append({
                'generation': generation,
                'best_fitness': current_best.fitness,
                'best_win_rate': current_best.win_rate,
                'best_profit_factor': current_best.profit_factor,
                'best_trade_count': current_best.trade_count,
                'best_genes': current_best.to_dict()
            })

            if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
                self.best_solution = current_best.copy()

            # 打印进度
            if generation % 5 == 0 or generation == NUM_GENERATIONS - 1:
                stats = self.population.get_statistics()
                print(f"\nGen {generation}: Best Fitness={stats['best_fitness']:.4f}, "
                      f"Win Rate={stats['best_win_rate']:.2%}, "
                      f"Avg Fitness={stats['avg_fitness']:.4f}")

            # 进化
            self.population.evolve()

            # 评估新一代
            evaluate_population(self.df, self.population)

        # 最终结果
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)

        if self.best_solution:
            print(f"\nBest Solution Found:")
            print(f"  Fitness: {self.best_solution.fitness:.4f}")
            print(f"  Win Rate: {self.best_solution.win_rate:.2%}")
            print(f"  Profit Factor: {self.best_solution.profit_factor:.2f}")
            print(f"  Max Drawdown: {self.best_solution.max_drawdown:.4f}")
            print(f"  Trade Count: {self.best_solution.trade_count}")
            print(f"\nBest Parameters:")
            for key, value in self.best_solution.to_dict().items():
                print(f"  {key}: {value}")

        return self.best_solution, self.history

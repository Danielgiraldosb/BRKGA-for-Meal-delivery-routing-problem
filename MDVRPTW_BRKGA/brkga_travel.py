"""
BRKGA for MDVRPTW - Travel Time Minimization
Implements Biased Random-Key Genetic Algorithm to minimize average travel time
"""

import numpy as np
import time as time_module
import os
from utils import load_instance, save_results
from operators import BRKGAParameters, select_elite, biased_crossover, generate_mutant
from decoder import decode_chromosome


class BRKGATravel:
    """
    BRKGA Solver for MDVRPTW with travel time minimization objective
    """

    def __init__(self, orders, couriers, stores, instance_info, params=None):
        """
        Initialize BRKGA solver

        Args:
            orders: Dictionary of order data
            couriers: Dictionary of courier data
            stores: Dictionary of store data
            instance_info: Dictionary with instance metadata
            params: BRKGAParameters instance (or None for defaults)
        """
        self.orders = orders
        self.couriers = couriers
        self.stores = stores
        self.instance_info = instance_info

        # Set parameters
        if params is None:
            self.params = BRKGAParameters()
        else:
            self.params = params

        # Chromosome dimensions
        self.n_orders = len(orders)
        self.n_couriers = len(couriers)
        self.chromosome_length = self.n_orders + self.n_couriers

        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        self.generation_times = []

        print(f"\nBRKGA-Travel Initialized")
        print(f"Instance: {instance_info['name']}")
        print(f"Orders: {self.n_orders}, Couriers: {self.n_couriers}, Stores: {instance_info['n_stores']}")
        print(f"Chromosome length: {self.chromosome_length}")
        print(self.params)

    def solve(self):
        """
        Run BRKGA algorithm to find best solution

        Returns:
            Best solution dictionary
        """
        start_time = time_module.time()

        # Initialize population
        if self.params.verbose:
            print("\nInitializing population...")

        population = np.random.uniform(
            0, 1,
            (self.params.population_size, self.chromosome_length)
        )

        generations_without_improvement = 0
        best_generation = 0

        # Evolution loop
        for generation in range(self.params.max_generations):
            gen_start_time = time_module.time()

            # Evaluate population
            solutions = []
            fitness_values = np.zeros(self.params.population_size)

            for i in range(self.params.population_size):
                solution = decode_chromosome(
                    population[i], self.orders, self.couriers, self.stores,
                    self.params, fitness_type='travel'
                )
                solutions.append(solution)
                fitness_values[i] = solution['fitness']

            # Track best solution in this generation
            gen_best_idx = np.argmin(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            gen_best_solution = solutions[gen_best_idx]

            # Update global best
            if gen_best_fitness < self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = gen_best_solution
                generations_without_improvement = 0
                best_generation = generation

                if self.params.verbose:
                    print(f"\nGen {generation}: NEW BEST")
                    print(f"  Fitness: {self.best_fitness:.4f}")
                    print(f"  Avg travel: {self.best_solution['avg_travel_per_order']:.2f} min/order")
                    print(f"  Coverage: {self.best_solution['orders_covered']}/{self.n_orders} "
                          f"({self.best_solution['coverage_rate']*100:.2f}%)")
                    print(f"  Total travel: {self.best_solution['total_travel_time']:.2f} min")
            else:
                generations_without_improvement += 1

            self.convergence_history.append(self.best_fitness)
            gen_time = time_module.time() - gen_start_time
            self.generation_times.append(gen_time)

            # Progress update (every 10 generations)
            if self.params.verbose and generation % 10 == 0:
                print(f"Gen {generation}: Best fitness = {self.best_fitness:.4f}, "
                      f"Avg travel = {self.best_solution['avg_travel_per_order']:.2f} min, "
                      f"Time = {gen_time:.2f}s")

            # Early stopping
            if generations_without_improvement >= self.params.early_stop_patience:
                if self.params.verbose:
                    print(f"\nEarly stopping at generation {generation}")
                    print(f"No improvement for {self.params.early_stop_patience} generations")
                break

            # Selection
            elite_pop, elite_indices = select_elite(
                population, fitness_values, self.params.elite_size
            )
            non_elite_indices = [i for i in range(self.params.population_size)
                                if i not in elite_indices]

            # Build next generation
            next_population = []

            # 1. Preserve elite
            next_population.extend(elite_pop)

            # 2. Generate offspring via biased crossover
            for _ in range(self.params.offspring_size):
                # Select parents
                elite_parent_idx = np.random.choice(elite_indices)
                non_elite_parent_idx = np.random.choice(non_elite_indices)

                elite_parent = population[elite_parent_idx]
                non_elite_parent = population[non_elite_parent_idx]

                # Crossover
                offspring = biased_crossover(
                    elite_parent, non_elite_parent, self.params.elite_bias
                )
                next_population.append(offspring)

            # 3. Add mutants
            for _ in range(self.params.mutant_size):
                mutant = generate_mutant(self.chromosome_length)
                next_population.append(mutant)

            population = np.array(next_population)

        # Final summary
        total_time = time_module.time() - start_time
        avg_gen_time = np.mean(self.generation_times)

        print(f"\n{'='*60}")
        print(f"BRKGA-Travel FINISHED")
        print(f"{'='*60}")
        print(f"Total generations: {generation + 1}")
        print(f"Best generation: {best_generation}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Avg time per generation: {avg_gen_time:.2f}s")
        print(f"\nBEST SOLUTION:")
        print(f"  Fitness: {self.best_fitness:.4f}")
        print(f"  Avg travel per order: {self.best_solution['avg_travel_per_order']:.2f} min")
        print(f"  Total travel time: {self.best_solution['total_travel_time']:.2f} min")
        print(f"  Coverage: {self.best_solution['orders_covered']}/{self.n_orders} "
              f"({self.best_solution['coverage_rate']*100:.2f}%)")
        print(f"  Routes created: {len(self.best_solution['assignments'])}")
        print(f"{'='*60}\n")

        return self.best_solution

    def save_solution(self, output_folder='results'):
        """
        Save best solution to Excel file

        Args:
            output_folder: Output directory
        """
        if self.best_solution is None:
            print("No solution to save. Run solve() first.")
            return

        output_dir = os.path.join('MDVRPTW_BRKGA', output_folder, 'travel')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f'resultadosIns_{self.instance_info["name"]}_travel.xlsx'
        )

        # Update courier data with final state
        final_couriers = self.best_solution['solution_state']['couriers']

        save_results(self.best_solution, output_path, final_couriers)

        print(f"Solution saved to: {output_path}")


def main():
    """
    Main execution function - run BRKGA-Travel on specified instance
    """
    import argparse

    parser = argparse.ArgumentParser(description='BRKGA for MDVRPTW - Travel Time Minimization')
    parser.add_argument('--instance', type=str, default='Instancia_0',
                       help='Instance name (e.g., Instancia_0, Instancia_18)')
    parser.add_argument('--pop_size', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=500,
                       help='Maximum generations')
    parser.add_argument('--elite_size', type=int, default=20,
                       help='Elite size')
    parser.add_argument('--mutant_size', type=int, default=15,
                       help='Mutant size')
    parser.add_argument('--elite_bias', type=float, default=0.7,
                       help='Elite bias (0-1)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"BRKGA-Travel for MDVRPTW")
    print(f"{'='*60}")
    print(f"Loading instance: {args.instance}")

    # Load instance
    orders, couriers, stores, instance_info = load_instance(args.instance)

    print(f"Loaded: {len(orders)} orders, {len(couriers)} couriers, {len(stores)} stores")

    # Set parameters
    params = BRKGAParameters(
        population_size=args.pop_size,
        elite_size=args.elite_size,
        mutant_size=args.mutant_size,
        elite_bias=args.elite_bias,
        max_generations=args.generations,
        early_stop_patience=args.patience,
        verbose=True
    )

    # Create solver and run
    solver = BRKGATravel(orders, couriers, stores, instance_info, params)
    solution = solver.solve()

    # Save results
    solver.save_solution()

    return solution


if __name__ == '__main__':
    main()

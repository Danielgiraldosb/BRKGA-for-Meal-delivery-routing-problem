"""
BRKGA Genetic Operators
Implements elite selection, biased crossover, and mutation for BRKGA
"""

import numpy as np


def select_elite(population, fitness_values, elite_size):
    """
    Select elite individuals from population based on fitness

    Args:
        population: numpy array of shape (pop_size, chromosome_length)
        fitness_values: numpy array of fitness scores (lower is better)
        elite_size: Number of elite individuals to select

    Returns:
        Tuple of (elite_population, elite_indices)
    """
    # Sort by fitness (ascending: lower fitness = better solution)
    sorted_indices = np.argsort(fitness_values)
    elite_indices = sorted_indices[:elite_size]
    elite_population = population[elite_indices]

    return elite_population, elite_indices


def biased_crossover(elite_parent, non_elite_parent, elite_bias=0.7):
    """
    Perform biased random-key crossover between two parents

    For each gene position:
    - With probability elite_bias: inherit from elite parent
    - Otherwise: inherit from non-elite parent

    Args:
        elite_parent: numpy array (chromosome from elite set)
        non_elite_parent: numpy array (chromosome from non-elite set)
        elite_bias: Probability of inheriting from elite parent (default 0.7)

    Returns:
        numpy array (offspring chromosome)
    """
    chromosome_length = len(elite_parent)
    offspring = np.zeros(chromosome_length)

    # Vectorized version for efficiency
    inherit_from_elite = np.random.random(chromosome_length) < elite_bias
    offspring = np.where(inherit_from_elite, elite_parent, non_elite_parent)

    return offspring


def generate_mutant(chromosome_length):
    """
    Generate a completely random chromosome (mutant)

    Args:
        chromosome_length: Length of chromosome

    Returns:
        numpy array with random values in [0, 1]
    """
    return np.random.uniform(0, 1, chromosome_length)


def tournament_selection(population, fitness_values, tournament_size=3):
    """
    Select parent using tournament selection (optional enhancement)

    Args:
        population: numpy array of shape (pop_size, chromosome_length)
        fitness_values: numpy array of fitness scores
        tournament_size: Number of individuals in tournament

    Returns:
        Selected parent chromosome
    """
    pop_size = len(population)
    # Select random individuals for tournament
    tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
    tournament_fitness = fitness_values[tournament_indices]

    # Winner is individual with best (lowest) fitness
    winner_idx = tournament_indices[np.argmin(tournament_fitness)]
    return population[winner_idx]


class BRKGAParameters:
    """
    BRKGA parameter configuration
    """
    def __init__(
        self,
        population_size=100,
        elite_size=20,
        mutant_size=15,
        elite_bias=0.7,
        max_generations=500,
        early_stop_patience=100,
        time_step=2.0,
        max_batch_size=3,
        top_k_couriers=5,
        verbose=True
    ):
        """
        Initialize BRKGA parameters

        Args:
            population_size: Total number of individuals
            elite_size: Number of elite individuals (best solutions)
            mutant_size: Number of random mutants per generation
            elite_bias: Probability of inheriting gene from elite parent (0.7 = 70%)
            max_generations: Maximum number of generations
            early_stop_patience: Stop if no improvement for this many generations
            time_step: Time step for simulation (minutes)
            max_batch_size: Maximum orders per route
            top_k_couriers: Number of top couriers to evaluate per batch
            verbose: Print progress messages
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutant_size = mutant_size
        self.elite_bias = elite_bias
        self.max_generations = max_generations
        self.early_stop_patience = early_stop_patience
        self.time_step = time_step
        self.max_batch_size = max_batch_size
        self.top_k_couriers = top_k_couriers
        self.verbose = verbose

        # Validate parameters
        assert elite_size + mutant_size < population_size, \
            "Elite + mutant size must be less than population size"
        assert 0 < elite_bias < 1, "Elite bias must be between 0 and 1"

        # Calculated values
        self.offspring_size = population_size - elite_size - mutant_size

    def __repr__(self):
        return (
            f"BRKGAParameters(\n"
            f"  population_size={self.population_size},\n"
            f"  elite_size={self.elite_size},\n"
            f"  mutant_size={self.mutant_size},\n"
            f"  offspring_size={self.offspring_size},\n"
            f"  elite_bias={self.elite_bias},\n"
            f"  max_generations={self.max_generations},\n"
            f"  early_stop_patience={self.early_stop_patience}\n"
            f")"
        )

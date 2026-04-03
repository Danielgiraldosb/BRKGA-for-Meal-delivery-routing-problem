"""
Run BRKGA experiments on all instances
Compares BRKGA-Coverage and BRKGA-Travel across multiple instances
"""

import os
import pandas as pd
import numpy as np
import time as time_module
from glob import glob

from utils import load_instance
from operators import BRKGAParameters
from brkga_coverage import BRKGACoverage
from brkga_travel import BRKGATravel


def get_all_instances(base_folder=None):
    """
    Get list of all available instances

    Args:
        base_folder: Base folder containing instance directories (default: ../Instancias)

    Returns:
        List of instance names
    """
    # Set default base folder to parent directory
    if base_folder is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_folder = os.path.join(os.path.dirname(script_dir), 'Instancias')

    if not os.path.exists(base_folder):
        print(f"Error: {base_folder} not found!")
        return []

    # Find all subdirectories starting with 'Instancia'
    instance_dirs = glob(os.path.join(base_folder, 'Instancia_*'))
    instances = [os.path.basename(d) for d in instance_dirs]
    instances.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

    return instances


def run_single_experiment(instance_name, algorithm='coverage', params=None):
    """
    Run single BRKGA experiment on specified instance

    Args:
        instance_name: Name of instance
        algorithm: 'coverage' or 'travel'
        params: BRKGAParameters instance

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"Running BRKGA-{algorithm.upper()} on {instance_name}")
    print(f"{'='*70}")

    start_time = time_module.time()

    # Load instance
    try:
        orders, couriers, stores, instance_info = load_instance(instance_name)
    except Exception as e:
        print(f"Error loading instance: {e}")
        return None

    # Create solver
    if algorithm == 'coverage':
        solver = BRKGACoverage(orders, couriers, stores, instance_info, params)
    else:  # travel
        solver = BRKGATravel(orders, couriers, stores, instance_info, params)

    # Run solver
    try:
        solution = solver.solve()
        solver.save_solution()
    except Exception as e:
        print(f"Error during solving: {e}")
        return None

    end_time = time_module.time()
    total_time = end_time - start_time

    # Collect results
    result = {
        'instance': instance_name,
        'algorithm': algorithm,
        'n_orders': len(orders),
        'n_couriers': len(couriers),
        'n_stores': len(stores),
        'orders_covered': solution['orders_covered'],
        'coverage_rate': solution['coverage_rate'],
        'total_travel_time': solution['total_travel_time'],
        'avg_travel_per_order': solution['avg_travel_per_order'],
        'n_routes': len(solution['assignments']),
        'fitness': solution['fitness'],
        'generations': len(solver.convergence_history),
        'total_time_sec': total_time,
        'total_time_min': total_time / 60.0,
        'avg_time_per_generation': np.mean(solver.generation_times)
    }

    return result


def run_all_experiments(instances=None, algorithms=['coverage', 'travel'], params=None):
    """
    Run experiments on all instances with both algorithms

    Args:
        instances: List of instance names (or None for all)
        algorithms: List of algorithms to run
        params: BRKGAParameters instance

    Returns:
        DataFrame with all results
    """
    if instances is None:
        instances = get_all_instances()

    if not instances:
        print("No instances found!")
        return None

    print(f"\nFound {len(instances)} instances: {instances}")
    print(f"Running algorithms: {algorithms}")

    results = []

    for instance in instances:
        for algorithm in algorithms:
            print(f"\n{'#'*70}")
            print(f"Experiment: {instance} - {algorithm.upper()}")
            print(f"{'#'*70}")

            result = run_single_experiment(instance, algorithm, params)

            if result is not None:
                results.append(result)

            # Save intermediate results
            if results:
                df_results = pd.DataFrame(results)
                output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'experiment_results_temp.xlsx')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df_results.to_excel(output_path, index=False)
                print(f"\nIntermediate results saved to: {output_path}")

    # Create final results dataframe
    df_results = pd.DataFrame(results)

    return df_results


def generate_comparison_report(df_results):
    """
    Generate comparison report from experiment results

    Args:
        df_results: DataFrame with experiment results

    Returns:
        DataFrame with comparison statistics
    """
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}")

    # Pivot table for coverage comparison
    coverage_pivot = df_results.pivot_table(
        index='instance',
        columns='algorithm',
        values='coverage_rate',
        aggfunc='mean'
    )

    # Pivot table for travel time comparison
    travel_pivot = df_results.pivot_table(
        index='instance',
        columns='algorithm',
        values='avg_travel_per_order',
        aggfunc='mean'
    )

    # Summary statistics
    summary_stats = df_results.groupby('algorithm').agg({
        'coverage_rate': ['mean', 'std', 'min', 'max'],
        'avg_travel_per_order': ['mean', 'std', 'min', 'max'],
        'total_time_min': ['mean', 'std', 'min', 'max']
    }).round(4)

    print("\nSUMMARY STATISTICS:")
    print(summary_stats)

    print("\nCOVERAGE COMPARISON (%):")
    print((coverage_pivot * 100).round(2))

    print("\nTRAVEL TIME COMPARISON (min/order):")
    print(travel_pivot.round(2))

    # Save detailed report
    output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'comparison_report.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Results', index=False)
        coverage_pivot.to_excel(writer, sheet_name='Coverage Comparison')
        travel_pivot.to_excel(writer, sheet_name='Travel Comparison')
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')

    print(f"\nComparison report saved to: {output_path}")

    return summary_stats


def main():
    """
    Main execution function
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run BRKGA experiments on all instances')
    parser.add_argument('--instances', type=str, nargs='+', default=None,
                       help='Specific instances to run (e.g., Instancia_0 Instancia_1). If not specified, runs all.')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       choices=['coverage', 'travel', 'both'], default=['both'],
                       help='Which algorithms to run')
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

    # Determine which algorithms to run
    if 'both' in args.algorithms:
        algorithms = ['coverage', 'travel']
    else:
        algorithms = args.algorithms

    print(f"\n{'#'*70}")
    print("BRKGA EXPERIMENTS - ALL INSTANCES")
    print(f"{'#'*70}")

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

    print(f"\nParameters:")
    print(params)

    # Run experiments
    df_results = run_all_experiments(args.instances, algorithms, params)

    if df_results is None or df_results.empty:
        print("No results to report!")
        return

    # Save final results
    output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'experiment_results_final.xlsx')
    df_results.to_excel(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")

    # Generate comparison report
    summary = generate_comparison_report(df_results)

    print(f"\n{'#'*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()

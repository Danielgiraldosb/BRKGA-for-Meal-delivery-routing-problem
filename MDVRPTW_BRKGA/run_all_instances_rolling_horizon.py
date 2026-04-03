"""
Run BRKGA Rolling Horizon on all instances
Executes both coverage and travel fitness types
"""

import os
import pandas as pd
import time as time_module
from glob import glob

from utils import load_instance
from operators import BRKGAParameters
from brkga_rolling_horizon import BRKGARollingHorizon


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
        print("Error: {} not found!".format(base_folder))
        return []

    # Find all subdirectories starting with 'Instancia'
    instance_dirs = glob(os.path.join(base_folder, 'Instancia_*'))
    instances = [os.path.basename(d) for d in instance_dirs]
    instances.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

    return instances


def run_single_instance(instance_name, fitness_type='coverage', params=None):
    """
    Run BRKGA Rolling Horizon on a single instance

    Args:
        instance_name: Name of instance (e.g., 'Instancia_0')
        fitness_type: 'coverage' or 'travel'
        params: BRKGAParameters instance

    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print("Running BRKGA Rolling Horizon - {} on {}".format(fitness_type.upper(), instance_name))
    print("="*70)

    start_time = time_module.time()

    # Load instance
    try:
        orders, couriers, stores, instance_info = load_instance(instance_name)
        print("Instance loaded successfully:")
        print("  - Orders: {}".format(len(orders)))
        print("  - Couriers: {}".format(len(couriers)))
        print("  - Stores: {}".format(len(stores)))
    except Exception as e:
        print("Error loading instance: {}".format(e))
        return None

    # Create solver
    try:
        solver = BRKGARollingHorizon(
            orders, couriers, stores, instance_info, params, fitness_type
        )
    except Exception as e:
        print("Error creating solver: {}".format(e))
        return None

    # Run solver
    try:
        solution = solver.solve()
        solver.save_solution(solution)
    except Exception as e:
        print("Error during solving: {}".format(e))
        import traceback
        traceback.print_exc()
        return None

    end_time = time_module.time()
    total_time = end_time - start_time

    # Collect results
    result = {
        'instance': instance_name,
        'fitness_type': fitness_type,
        'n_orders': len(orders),
        'n_couriers': len(couriers),
        'n_stores': len(stores),
        'orders_covered': solution['orders_covered'],
        'coverage_rate': solution['coverage_rate'],
        'total_travel_time': solution['total_travel_time'],
        'avg_travel_per_order': solution['avg_travel_per_order'],
        'n_routes': len(solution['assignments']),
        'window_count': solution['window_count'],
        'brkga_executions': solution['brkga_executions'],
        'execution_time_sec': solution['execution_time_seconds'],
        'total_runtime_sec': total_time
    }

    print("\n" + "="*70)
    print("COMPLETED: {} - {}".format(instance_name, fitness_type.upper()))
    print("  Coverage: {:.2f}%".format(result['coverage_rate'] * 100))
    print("  Avg travel: {:.2f} min/order".format(result['avg_travel_per_order']))
    print("  Execution time: {:.2f} seconds".format(result['execution_time_sec']))
    print("="*70)

    return result


def run_all_instances(instances=None, fitness_types=['coverage', 'travel'], params=None):
    """
    Run BRKGA Rolling Horizon on all instances

    Args:
        instances: List of instance names (or None for all)
        fitness_types: List of fitness types to run
        params: BRKGAParameters instance

    Returns:
        DataFrame with all results
    """
    if instances is None:
        instances = get_all_instances()

    if not instances:
        print("No instances found!")
        return None

    print("\n" + "#"*70)
    print("BRKGA ROLLING HORIZON - ALL INSTANCES")
    print("#"*70)
    print("Found {} instances: {}".format(len(instances), instances))
    print("Running fitness types: {}".format(fitness_types))

    results = []

    for instance in instances:
        for fitness_type in fitness_types:
            print("\n" + "#"*70)
            print("Experiment: {} - {}".format(instance, fitness_type.upper()))
            print("#"*70)

            result = run_single_instance(instance, fitness_type, params)

            if result is not None:
                results.append(result)

            # Save intermediate results
            if results:
                df_results = pd.DataFrame(results)
                output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'all_instances_results_temp.xlsx')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df_results.to_excel(output_path, index=False)
                print("\nIntermediate results saved to: {}".format(output_path))

    # Create final results dataframe
    if results:
        df_results = pd.DataFrame(results)
        return df_results
    else:
        return None


def generate_summary_report(df_results):
    """
    Generate summary report from results

    Args:
        df_results: DataFrame with experiment results

    Returns:
        Summary statistics dataframe
    """
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    # Pivot table for coverage comparison
    coverage_pivot = df_results.pivot_table(
        index='instance',
        columns='fitness_type',
        values='coverage_rate',
        aggfunc='mean'
    )

    # Pivot table for travel time comparison
    travel_pivot = df_results.pivot_table(
        index='instance',
        columns='fitness_type',
        values='avg_travel_per_order',
        aggfunc='mean'
    )

    # Summary statistics by fitness type
    summary_stats = df_results.groupby('fitness_type').agg({
        'coverage_rate': ['mean', 'std', 'min', 'max'],
        'avg_travel_per_order': ['mean', 'std', 'min', 'max'],
        'execution_time_sec': ['mean', 'std', 'min', 'max']
    }).round(4)

    print("\nSUMMARY STATISTICS:")
    print(summary_stats)

    print("\nCOVERAGE COMPARISON (%):")
    print((coverage_pivot * 100).round(2))

    print("\nTRAVEL TIME COMPARISON (min/order):")
    print(travel_pivot.round(2))

    # Save detailed report
    output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'rolling_horizon_summary_report.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Results', index=False)
        coverage_pivot.to_excel(writer, sheet_name='Coverage Comparison')
        travel_pivot.to_excel(writer, sheet_name='Travel Comparison')
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')

    print("\nSummary report saved to: {}".format(output_path))

    return summary_stats


def main():
    """
    Main execution function
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Run BRKGA Rolling Horizon on all instances'
    )
    parser.add_argument('--instances', type=str, nargs='+', default=None,
                       help='Specific instances to run (e.g., Instancia_0 Instancia_1). If not specified, runs all.')
    parser.add_argument('--fitness', type=str, nargs='+',
                       choices=['coverage', 'travel', 'both'], default=['both'],
                       help='Which fitness types to run')
    parser.add_argument('--pop_size', type=int, default=10,
                       help='Population size per window')
    parser.add_argument('--generations', type=int, default=10,
                       help='Generations per window')
    parser.add_argument('--elite_size', type=int, default=2,
                       help='Elite size')
    parser.add_argument('--mutant_size', type=int, default=2,
                       help='Mutant size')
    parser.add_argument('--elite_bias', type=float, default=0.7,
                       help='Elite bias (0-1)')

    args = parser.parse_args()

    # Determine which fitness types to run
    if 'both' in args.fitness:
        fitness_types = ['coverage', 'travel']
    else:
        fitness_types = args.fitness

    print("\n" + "#"*70)
    print("BRKGA ROLLING HORIZON EXPERIMENTS - ALL INSTANCES")
    print("#"*70)

    # Set parameters
    params = BRKGAParameters(
        population_size=args.pop_size,
        elite_size=args.elite_size,
        mutant_size=args.mutant_size,
        elite_bias=args.elite_bias,
        max_generations=args.generations,
        early_stop_patience=args.generations,  # No early stop in windows
        verbose=False
    )

    print("\nParameters per window:")
    print("  Population size: {}".format(params.population_size))
    print("  Elite size: {}".format(params.elite_size))
    print("  Mutant size: {}".format(params.mutant_size))
    print("  Generations: {}".format(params.max_generations))

    # Run experiments
    df_results = run_all_instances(args.instances, fitness_types, params)

    if df_results is None or df_results.empty:
        print("No results to report!")
        return

    # Save final results
    output_path = os.path.join('MDVRPTW_BRKGA', 'results', 'rolling_horizon_all_instances_final.xlsx')
    df_results.to_excel(output_path, index=False)
    print("\nFinal results saved to: {}".format(output_path))

    # Generate summary report
    summary = generate_summary_report(df_results)

    print("\n" + "#"*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("#"*70)


if __name__ == '__main__':
    main()

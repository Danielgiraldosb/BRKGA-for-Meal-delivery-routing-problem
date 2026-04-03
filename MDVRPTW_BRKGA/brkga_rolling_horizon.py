"""
BRKGA con Rolling Horizon para MDVRPTW
Comparación justa con método Greedy de MDRPcorporativo.ipynb

Ejecuta BRKGA cada 2 minutos viendo solo órdenes disponibles en ese momento,
sin conocimiento del futuro.
"""

import numpy as np
import time as time_module
import os
import copy
from utils import (
    load_instance, save_results, travel_time_min,
    initialize_solution_state, group_orders_by_store,
    get_available_couriers, release_finished_couriers
)
from operators import BRKGAParameters, select_elite, biased_crossover, generate_mutant
from decoder import nearest_neighbor_sequence, fitness_coverage, fitness_travel


class BRKGARollingHorizon:
    """
    BRKGA con Rolling Horizon de 2 minutos
    Solo ve órdenes disponibles en cada paso temporal
    """

    def __init__(self, orders, couriers, stores, instance_info, params=None, fitness_type='coverage'):
        """
        Initialize BRKGA Rolling Horizon solver

        Args:
            orders: Dictionary of order data
            couriers: Dictionary of courier data
            stores: Dictionary of store data
            instance_info: Dictionary with instance metadata
            params: BRKGAParameters instance (or None for defaults)
            fitness_type: 'coverage' or 'travel'
        """
        self.orders = orders
        self.couriers = couriers
        self.stores = stores
        self.instance_info = instance_info
        self.fitness_type = fitness_type

        # Parámetros BRKGA para cada ventana de tiempo
        if params is None:
            self.params = BRKGAParameters(
                population_size=10,
                elite_size=2,
                mutant_size=2,
                elite_bias=0.7,
                max_generations=10,
                early_stop_patience=5,
                verbose=False  # No verbose para cada ventana
            )
        else:
            self.params = params

        print(f"\n{'='*60}")
        print(f"BRKGA Rolling Horizon - {fitness_type.upper()}")
        print(f"{'='*60}")
        print(f"Instance: {instance_info['name']}")
        print(f"Orders: {len(orders)}, Couriers: {len(couriers)}, Stores: {len(stores)}")
        print(f"Window: 2 minutes (matching MDRPcorporativo)")
        print(f"BRKGA params per window: pop={self.params.population_size}, "
              f"gen={self.params.max_generations}")

    def solve(self):
        """
        Run BRKGA with Rolling Horizon (2-minute windows)

        Returns:
            Complete solution dictionary
        """
        start_time = time_module.time()

        # Initialize state
        solution_state = initialize_solution_state(self.couriers)
        all_assignments = []
        total_travel_time = 0.0
        orders_assigned = set()

        # Time window parameters (matching MDRPcorporativo)
        simulation_start = 0.0
        simulation_end = 1440.0
        time_step = 2.0  # 2 minutos

        current_time = simulation_start
        window_count = 0
        brkga_executions = 0

        print(f"\nStarting Rolling Horizon simulation...")
        print(f"Time: 0 to 1440 min, step = 2 min")

        # Rolling Horizon Loop
        while current_time <= simulation_end:

            # Release finished couriers
            release_finished_couriers(solution_state, current_time)

            # Get ready orders at THIS moment only
            ready_orders = [
                oid for oid in self.orders.keys()
                if self.orders[oid]['placement_min'] is not None
                and self.orders[oid]['placement_min'] <= current_time
                and oid not in orders_assigned
            ]

            if not ready_orders:
                current_time += time_step
                continue

            # Get available couriers at THIS moment
            available_couriers = get_available_couriers(
                solution_state, current_time, self.couriers
            )

            if not available_couriers:
                current_time += time_step
                continue

            # Execute BRKGA for THIS time window only
            window_assignments = self._optimize_window(
                ready_orders,
                available_couriers,
                solution_state,
                current_time
            )

            brkga_executions += 1

            # Apply assignments from this window
            for assignment in window_assignments:
                all_assignments.append(assignment)
                total_travel_time += assignment['total_travel_min']
                orders_assigned.update(assignment['orders'])

                # Update state
                courier_id = assignment['courier_id']
                solution_state['couriers'][courier_id]['status'] = 'busy'
                solution_state['couriers'][courier_id]['available_at'] = \
                    assignment['finish_min']
                solution_state['couriers'][courier_id]['assigned_routes'].append({
                    'time_depart_min': current_time,
                    'store_id': assignment['store_id'],
                    'orders': assignment['orders'],
                    'finish_min': assignment['finish_min'],
                    'total_travel_min': assignment['total_travel_min']
                })

            window_count += 1

            # Progress update every 100 minutes
            if window_count % 50 == 0:
                elapsed = time_module.time() - start_time
                print(f"  Time {current_time:.0f} min: {len(orders_assigned)} orders assigned, "
                      f"{brkga_executions} BRKGA runs, {elapsed:.1f}s elapsed")

            current_time += time_step

        # Final statistics
        total_time = time_module.time() - start_time
        orders_covered = len(orders_assigned)
        coverage_rate = orders_covered / len(self.orders)

        print(f"\n{'='*60}")
        print(f"BRKGA Rolling Horizon FINISHED")
        print(f"{'='*60}")
        print(f"Total time windows: {window_count}")
        print(f"BRKGA executions: {brkga_executions}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Avg time per window: {total_time/window_count:.3f}s")
        print(f"\nBEST SOLUTION:")
        print(f"  Orders covered: {orders_covered}/{len(self.orders)} ({coverage_rate*100:.2f}%)")
        print(f"  Routes created: {len(all_assignments)}")
        print(f"  Total travel time: {total_travel_time:.2f} min")
        print(f"  Avg travel per order: {total_travel_time/max(orders_covered,1):.2f} min")
        print(f"{'='*60}\n")

        return {
            'assignments': all_assignments,
            'total_travel_time': total_travel_time,
            'orders_covered': orders_covered,
            'total_orders': len(self.orders),
            'coverage_rate': coverage_rate,
            'avg_travel_per_order': total_travel_time / max(orders_covered, 1),
            'solution_state': solution_state,
            'window_count': window_count,
            'brkga_executions': brkga_executions,
            'execution_time_seconds': total_time
        }

    def _optimize_window(self, ready_order_ids, available_courier_ids,
                         solution_state, current_time):
        """
        Optimize assignments for current time window using BRKGA

        Args:
            ready_order_ids: List of ready order IDs
            available_courier_ids: List of available courier IDs
            solution_state: Current state
            current_time: Current simulation time

        Returns:
            List of assignments for this window
        """
        # Create mini problem for this window
        n_orders = len(ready_order_ids)
        n_couriers = len(available_courier_ids)

        if n_orders == 0 or n_couriers == 0:
            return []

        # Create chromosome mapping
        order_id_to_idx = {oid: i for i, oid in enumerate(ready_order_ids)}
        courier_id_to_idx = {cid: i for i, cid in enumerate(available_courier_ids)}

        chromosome_length = n_orders + n_couriers

        # Initialize population
        population = np.random.uniform(0, 1, (self.params.population_size, chromosome_length))

        best_assignments = []
        best_fitness = float('inf')

        # BRKGA evolution for this window
        for generation in range(self.params.max_generations):

            # Evaluate population
            solutions = []
            fitness_values = np.zeros(self.params.population_size)

            for i in range(self.params.population_size):
                assignments, fitness = self._decode_window_chromosome(
                    population[i],
                    ready_order_ids,
                    available_courier_ids,
                    solution_state,
                    current_time,
                    order_id_to_idx,
                    courier_id_to_idx
                )
                solutions.append(assignments)
                fitness_values[i] = fitness

            # Track best
            gen_best_idx = np.argmin(fitness_values)
            if fitness_values[gen_best_idx] < best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_assignments = solutions[gen_best_idx]

            # Selection and reproduction
            if generation < self.params.max_generations - 1:
                elite_pop, elite_indices = select_elite(
                    population, fitness_values, self.params.elite_size
                )
                non_elite_indices = [i for i in range(self.params.population_size)
                                    if i not in elite_indices]

                next_population = []
                next_population.extend(elite_pop)

                # Offspring
                offspring_count = self.params.population_size - self.params.elite_size - self.params.mutant_size
                for _ in range(offspring_count):
                    elite_parent_idx = np.random.choice(elite_indices)
                    non_elite_parent_idx = np.random.choice(non_elite_indices)
                    offspring = biased_crossover(
                        population[elite_parent_idx],
                        population[non_elite_parent_idx],
                        self.params.elite_bias
                    )
                    next_population.append(offspring)

                # Mutants
                for _ in range(self.params.mutant_size):
                    mutant = generate_mutant(chromosome_length)
                    next_population.append(mutant)

                population = np.array(next_population)

        return best_assignments

    def _decode_window_chromosome(self, chromosome, ready_order_ids, available_courier_ids,
                                   solution_state, current_time, order_id_to_idx, courier_id_to_idx):
        """
        Decode chromosome for current window

        Returns:
            Tuple of (assignments, fitness)
        """
        n_orders = len(ready_order_ids)
        n_couriers = len(available_courier_ids)

        order_keys = chromosome[:n_orders]
        courier_keys = chromosome[n_orders:]

        # Sort by priorities
        sorted_order_indices = np.argsort(-order_keys)
        sorted_orders = [ready_order_ids[i] for i in sorted_order_indices]

        courier_priority = {available_courier_ids[i]: courier_keys[i]
                           for i in range(n_couriers)}

        assignments = []
        assigned_orders = set()
        assigned_couriers = set()
        total_travel = 0.0

        # Group by store
        store_groups = {}
        for oid in sorted_orders:
            sid = self.orders[oid]['store_id']
            if sid not in store_groups:
                store_groups[sid] = []
            store_groups[sid].append(oid)

        # For each store
        for store_id, store_orders in store_groups.items():
            if store_id not in self.stores:
                continue

            store = self.stores[store_id]
            available = [cid for cid in available_courier_ids if cid not in assigned_couriers]

            if not available:
                break

            # Sort couriers by priority
            available.sort(key=lambda cid: courier_priority[cid], reverse=True)

            # Try batches
            while store_orders and available:
                max_batch = min(3, len(store_orders))

                best_assignment = None
                best_courier = None
                best_batch = None
                best_score = float('inf')

                for batch_size in range(1, max_batch + 1):
                    batch = store_orders[:batch_size]

                    # Try top K couriers (reduced for speed)
                    K = min(3, len(available))
                    for cid in available[:K]:
                        route = self._compute_route(
                            cid, store_id, batch, current_time, solution_state
                        )

                        if route is None:
                            continue

                        score = route['finish_min']
                        if score < best_score:
                            best_score = score
                            best_assignment = route
                            best_courier = cid
                            best_batch = batch

                if best_assignment:
                    assignments.append({
                        'time_depart_min': current_time,
                        'courier_id': best_courier,
                        'store_id': store_id,
                        'orders': best_batch,
                        'depart_store_min': best_assignment['depart_store_min'],
                        'deliveries': best_assignment['deliveries'],
                        'finish_min': best_assignment['finish_min'],
                        'total_travel_min': best_assignment['total_travel']
                    })

                    total_travel += best_assignment['total_travel']
                    assigned_orders.update(best_batch)
                    assigned_couriers.add(best_courier)

                    store_orders = [o for o in store_orders if o not in best_batch]
                    available.remove(best_courier)
                else:
                    break

        # Calculate fitness
        orders_covered = len(assigned_orders)

        if self.fitness_type == 'coverage':
            fitness = fitness_coverage(orders_covered, n_orders, total_travel)
        else:
            fitness = fitness_travel(orders_covered, n_orders, total_travel)

        return assignments, fitness

    def _compute_route(self, courier_id, store_id, order_batch, start_time, solution_state):
        """
        Compute route for courier-batch combination
        """
        courier = solution_state['couriers'][courier_id]
        store = self.stores[store_id]

        vehicle = courier['vehicle']
        current_lat = courier['current_lat']
        current_lng = courier['current_lng']

        # Travel to store
        travel_to_store = travel_time_min(
            current_lat, current_lng,
            store['pick_up_lat'], store['pick_up_lng'],
            vehicle
        )
        arrive_store = start_time + travel_to_store

        # Wait for orders
        group_ready_time = max([self.orders[oid]['ready_min'] for oid in order_batch])
        depart_store = max(arrive_store, group_ready_time)

        # Check courier shift
        if depart_store < courier['on_time_min'] or depart_store > courier['off_time_min']:
            return None

        # Nearest neighbor routing
        sequence, deliveries = nearest_neighbor_sequence(
            order_batch, store, self.orders, vehicle, depart_store
        )

        if sequence is None:
            return None

        # Return to base
        last_delivery = deliveries[-1]
        return_time = travel_time_min(
            last_delivery['drop_lat'], last_delivery['drop_lng'],
            courier['home_lat'], courier['home_lng'],
            vehicle
        )
        finish_time = last_delivery['drop_time_min'] + return_time

        if finish_time > courier['off_time_min']:
            return None

        total_travel = finish_time - start_time

        return {
            'sequence': sequence,
            'deliveries': deliveries,
            'depart_store_min': depart_store,
            'finish_min': finish_time,
            'total_travel': total_travel
        }

    def save_solution(self, solution, output_folder='results'):
        """
        Save solution to Excel file
        """
        output_dir = os.path.join('MDVRPTW_BRKGA', output_folder, f'rolling_horizon_{self.fitness_type}')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f'resultadosIns_{self.instance_info["name"]}_RH_{self.fitness_type}.xlsx'
        )

        final_couriers = solution['solution_state']['couriers']
        execution_time = solution.get('execution_time_seconds', None)
        save_results(solution, output_path, final_couriers, execution_time)

        print(f"Solution saved to: {output_path}")


def main():
    """
    Main execution - BRKGA Rolling Horizon
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='BRKGA Rolling Horizon for MDVRPTW (Fair comparison with Greedy)'
    )
    parser.add_argument('--instance', type=str, default='Instancia_4',
                       help='Instance name')
    parser.add_argument('--fitness', type=str, default='coverage',
                       choices=['coverage', 'travel'],
                       help='Fitness type')
    parser.add_argument('--pop_size', type=int, default=10,
                       help='Population size per window')
    parser.add_argument('--generations', type=int, default=10,
                       help='Generations per window')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"BRKGA Rolling Horizon - {args.fitness.upper()}")
    print(f"{'='*60}")
    print(f"Instance: {args.instance}")
    print(f"Fitness: {args.fitness}")

    # Load instance
    orders, couriers, stores, instance_info = load_instance(args.instance)

    # Parameters
    params = BRKGAParameters(
        population_size=args.pop_size,
        elite_size=max(2, args.pop_size // 5),
        mutant_size=max(2, args.pop_size // 5),
        elite_bias=0.7,
        max_generations=args.generations,
        early_stop_patience=args.generations,  # No early stop in windows
        verbose=False
    )

    # Solve
    solver = BRKGARollingHorizon(
        orders, couriers, stores, instance_info, params, args.fitness
    )
    solution = solver.solve()

    # Save
    solver.save_solution(solution)

    return solution


if __name__ == '__main__':
    main()

"""
Chromosome Decoder for MDVRPTW BRKGA
Converts random-key chromosomes into feasible routing solutions
"""

import numpy as np
import copy
from utils import (
    travel_time_min,
    initialize_solution_state,
    group_orders_by_store,
    get_available_couriers,
    release_finished_couriers
)


def decode_chromosome(chromosome, orders, couriers, stores, params, fitness_type='coverage'):
    """
    Convert random-key chromosome to feasible routing solution

    Args:
        chromosome: numpy array [order_keys, courier_keys]
        orders: Dictionary of order data
        couriers: Dictionary of courier data
        stores: Dictionary of store data
        params: BRKGAParameters instance
        fitness_type: 'coverage' or 'travel' (determines fitness function)

    Returns:
        Dictionary with solution details and fitness
    """
    # Extract chromosome segments
    n_orders = len(orders)
    n_couriers = len(couriers)

    order_keys = chromosome[:n_orders]
    courier_keys = chromosome[n_orders:]

    # Create priority mappings
    order_ids = list(orders.keys())
    courier_ids = list(couriers.keys())

    # Sort orders by their random keys (descending = higher priority first)
    sorted_order_indices = np.argsort(-order_keys)
    sorted_orders = [order_ids[i] for i in sorted_order_indices]

    # Map courier IDs to their priority keys
    courier_priority = {courier_ids[i]: courier_keys[i] for i in range(n_couriers)}

    # Initialize solution state
    solution_state = initialize_solution_state(couriers)
    assignments = []
    total_travel_time = 0.0
    orders_assigned = set()

    # Create mutable order assignment tracking
    order_assigned_flags = {oid: False for oid in orders.keys()}

    # Time-stepped simulation
    simulation_start = 0.0
    simulation_end = 1440.0
    time_step = params.time_step

    current_time = simulation_start

    while current_time <= simulation_end:
        # Release busy couriers who have finished
        release_finished_couriers(solution_state, current_time)

        # Get ready orders at current time (not yet assigned)
        ready_orders = [
            oid for oid in sorted_orders
            if orders[oid]['placement_min'] is not None
            and orders[oid]['placement_min'] <= current_time
            and not order_assigned_flags[oid]
        ]

        if not ready_orders:
            current_time += time_step
            continue

        # Group orders by store
        store_groups = group_orders_by_store(ready_orders, orders)

        # For each store with pending orders
        for store_id, store_order_list in store_groups.items():
            if store_id not in stores:
                continue

            store = stores[store_id]

            # Sort orders within store by chromosome priority
            store_order_list.sort(
                key=lambda oid: order_keys[order_ids.index(oid)],
                reverse=True
            )

            # Get available couriers at current time
            available_couriers = get_available_couriers(
                solution_state, current_time, couriers
            )

            if not available_couriers:
                break  # No couriers available

            # Sort available couriers by chromosome priority
            available_couriers.sort(
                key=lambda cid: courier_priority[cid],
                reverse=True
            )

            # Greedy batch assignment
            while store_order_list and available_couriers:
                best_assignment = None
                best_courier = None
                best_batch = None

                # Try batch sizes from 1 to min(max_batch_size, available orders)
                max_batch = min(params.max_batch_size, len(store_order_list))

                for batch_size in range(1, max_batch + 1):
                    batch = store_order_list[:batch_size]

                    # Try top K couriers (not all, for efficiency)
                    K_top_couriers = min(params.top_k_couriers, len(available_couriers))

                    for cid in available_couriers[:K_top_couriers]:
                        # Compute best route for this courier-batch
                        route = compute_best_route(
                            cid, store_id, batch,
                            current_time, orders, couriers, stores, solution_state
                        )

                        if route is None:  # Infeasible
                            continue

                        # Evaluate route quality (minimize finish time)
                        route_score = route['finish_min']

                        if (best_assignment is None or
                            route_score < best_assignment['finish_min']):
                            best_assignment = route
                            best_courier = cid
                            best_batch = batch

                # Assign best found route
                if best_assignment is not None:
                    # Update solution
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

                    total_travel_time += best_assignment['total_travel']
                    orders_assigned.update(best_batch)

                    # Mark orders as assigned
                    for oid in best_batch:
                        order_assigned_flags[oid] = True

                    # Mark courier as busy
                    solution_state['couriers'][best_courier]['status'] = 'busy'
                    solution_state['couriers'][best_courier]['available_at'] = \
                        best_assignment['finish_min']
                    solution_state['couriers'][best_courier]['assigned_routes'].append({
                        'time_depart_min': current_time,
                        'store_id': store_id,
                        'orders': best_batch,
                        'finish_min': best_assignment['finish_min'],
                        'total_travel_min': best_assignment['total_travel']
                    })

                    # Remove assigned orders and courier
                    store_order_list = [o for o in store_order_list if o not in best_batch]
                    available_couriers.remove(best_courier)
                else:
                    # No feasible assignment found
                    break

        current_time += time_step

    # Compute fitness
    orders_covered = len(orders_assigned)
    total_orders = len(orders)

    if fitness_type == 'coverage':
        fitness = fitness_coverage(orders_covered, total_orders, total_travel_time)
    else:  # 'travel'
        fitness = fitness_travel(orders_covered, total_orders, total_travel_time)

    return {
        'assignments': assignments,
        'total_travel_time': total_travel_time,
        'orders_covered': orders_covered,
        'total_orders': total_orders,
        'coverage_rate': orders_covered / total_orders,
        'avg_travel_per_order': total_travel_time / max(orders_covered, 1),
        'fitness': fitness,
        'solution_state': solution_state
    }


def compute_best_route(courier_id, store_id, order_batch, start_time,
                       orders, couriers, stores, solution_state):
    """
    Find best feasible route for courier to deliver order batch

    Uses nearest neighbor heuristic for delivery sequence

    Args:
        courier_id: Courier ID
        store_id: Store ID
        order_batch: List of order IDs
        start_time: Current simulation time
        orders: Dictionary of order data
        couriers: Dictionary of courier data (template)
        stores: Dictionary of store data
        solution_state: Current simulation state

    Returns:
        Route dictionary or None if infeasible
    """
    courier = solution_state['couriers'][courier_id]
    store = stores[store_id]

    vehicle = courier['vehicle']
    current_lat = courier['current_lat']
    current_lng = courier['current_lng']

    # Time to reach store
    travel_to_store = travel_time_min(
        current_lat, current_lng,
        store['pick_up_lat'], store['pick_up_lng'],
        vehicle
    )
    arrive_store = start_time + travel_to_store

    # Wait for all orders to be ready
    group_ready_time = max([orders[oid]['ready_min'] for oid in order_batch])
    depart_store = max(arrive_store, group_ready_time)

    # Check if courier shift allows this
    if depart_store < courier['on_time_min'] or depart_store > courier['off_time_min']:
        return None  # Infeasible

    # Nearest neighbor routing
    sequence, deliveries = nearest_neighbor_sequence(
        order_batch, store, orders, vehicle, depart_store
    )

    if sequence is None:
        return None  # Infeasible (time window violated)

    # Calculate return to home base
    last_delivery = deliveries[-1]
    return_time = travel_time_min(
        last_delivery['drop_lat'], last_delivery['drop_lng'],
        courier['home_lat'], courier['home_lng'],
        vehicle
    )
    finish_time = last_delivery['drop_time_min'] + return_time

    # Check if courier can finish within shift
    if finish_time > courier['off_time_min']:
        return None  # Infeasible

    total_travel = finish_time - start_time

    return {
        'sequence': sequence,
        'deliveries': deliveries,
        'depart_store_min': depart_store,
        'finish_min': finish_time,
        'total_travel': total_travel
    }


def nearest_neighbor_sequence(order_batch, store, orders, vehicle, depart_time):
    """
    Find delivery sequence using nearest neighbor heuristic

    Args:
        order_batch: List of order IDs
        store: Store dictionary
        orders: Dictionary of order data
        vehicle: Vehicle type
        depart_time: Time of departure from store

    Returns:
        Tuple of (sequence, deliveries) or (None, None) if infeasible
    """
    current_time = depart_time
    current_pos = (store['pick_up_lat'], store['pick_up_lng'])
    remaining = list(order_batch)
    sequence = []
    deliveries = []

    while remaining:
        # Find nearest unvisited order
        nearest_oid = None
        min_time = float('inf')

        for oid in remaining:
            order = orders[oid]
            tt = travel_time_min(
                current_pos[0], current_pos[1],
                order['drop_lat'], order['drop_lng'],
                vehicle
            )

            if tt < min_time:
                min_time = tt
                nearest_oid = oid

        # Visit nearest order
        order = orders[nearest_oid]
        current_time += min_time

        # Check time window feasibility
        if current_time > order['expected_drop_min']:
            return None, None  # Time window violated

        deliveries.append({
            'order_id': nearest_oid,
            'drop_time_min': current_time,
            'drop_lat': order['drop_lat'],
            'drop_lng': order['drop_lng']
        })

        sequence.append(nearest_oid)
        current_pos = (order['drop_lat'], order['drop_lng'])
        remaining.remove(nearest_oid)

    return sequence, deliveries


def fitness_coverage(orders_covered, total_orders, total_travel_time):
    """
    Fitness function for maximizing coverage

    Args:
        orders_covered: Number of orders assigned
        total_orders: Total number of orders
        total_travel_time: Total travel time across all routes

    Returns:
        Fitness value (lower is better)
    """
    if total_orders == 0:
        return 1e12

    coverage_rate = orders_covered / total_orders
    avg_travel = total_travel_time / max(orders_covered, 1)

    # Heavy penalty for low coverage
    if coverage_rate < 0.50:
        penalty = 1e9 * (1 - coverage_rate)
    elif coverage_rate < 0.60:
        penalty = 1e7 * (1 - coverage_rate)
    else:
        penalty = 1e5 * (1 - coverage_rate)

    # Secondary: normalize travel time (20 min baseline)
    fitness = penalty + (avg_travel / 20.0)

    return fitness


def fitness_travel(orders_covered, total_orders, total_travel_time):
    """
    Fitness function for minimizing travel time

    Args:
        orders_covered: Number of orders assigned
        total_orders: Total number of orders
        total_travel_time: Total travel time across all routes

    Returns:
        Fitness value (lower is better)
    """
    if total_orders == 0:
        return 1e12

    coverage_rate = orders_covered / total_orders
    avg_travel = total_travel_time / max(orders_covered, 1)

    # Moderate penalty for low coverage (not as strict)
    if coverage_rate < 0.50:
        penalty = 1e6 * (1 - coverage_rate)
    else:
        penalty = 1e4 * (1 - coverage_rate)

    # Primary: minimize average travel time per order
    fitness = penalty + avg_travel

    return fitness

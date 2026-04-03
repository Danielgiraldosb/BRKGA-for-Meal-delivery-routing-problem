"""
Utility functions for MDVRPTW BRKGA implementation
Extracted and adapted from MDRPcorporativo.ipynb
"""

import pandas as pd
import numpy as np
import os
from datetime import time
from math import radians, sin, cos, sqrt, atan2

# Vehicle speed mapping (km/h)
SPEED_MAP = {
    'motorcycle': 25.0,
    'bicycle': 20.0,
    'car': 15.0,
    'walking': 5.0
}


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def travel_time_min(lat1, lon1, lat2, lon2, vehicle_type):
    """
    Calculate travel time between two points for a given vehicle type

    Args:
        lat1, lon1: Start point coordinates
        lat2, lon2: End point coordinates
        vehicle_type: Type of vehicle ('motorcycle', 'bicycle', 'car', 'walking')

    Returns:
        Travel time in minutes
    """
    if pd.isnull(lat1) or pd.isnull(lat2) or pd.isnull(lon1) or pd.isnull(lon2):
        return 1e6
    sp = SPEED_MAP.get(str(vehicle_type).lower(), 15.0)
    dist = haversine_km(lat1, lon1, lat2, lon2)
    return (dist / sp) * 60.0


def time_to_min(t):
    """
    Convert time object to minutes since midnight

    Args:
        t: datetime.time object or compatible

    Returns:
        Minutes since midnight as float, or None if conversion fails
    """
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if isinstance(t, time):
        return t.hour*60 + t.minute + t.second/60.0
    try:
        tt = pd.to_datetime(t).time()
        return tt.hour*60 + tt.minute + tt.second/60.0
    except:
        return None


def to_time_obj(x):
    """
    Parse string/value to datetime.time object

    Args:
        x: Time string or value

    Returns:
        datetime.time object or None
    """
    try:
        return pd.to_datetime(x).time()
    except:
        return None


def load_instance(instance_name, base_folder=None):
    """
    Load orders, couriers, and stores data from instance folder

    Args:
        instance_name: Name of instance (e.g., 'Instancia_0', 'Instancia_18')
        base_folder: Base folder containing instances (default: ../Instancias)

    Returns:
        Tuple of (orders_dict, couriers_dict, stores_dict, instance_info)
    """
    # Set default base folder to parent directory
    if base_folder is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_folder = os.path.join(os.path.dirname(script_dir), 'Instancias')

    # Construct paths
    ruta_couriers = os.path.join(base_folder, instance_name, 'couriers.csv')
    ruta_stores = os.path.join(base_folder, instance_name, 'stores.csv')
    ruta_orders = os.path.join(base_folder, instance_name, 'orders.csv')

    # Read CSV files
    df_couriers = pd.read_csv(ruta_couriers)
    df_stores = pd.read_csv(ruta_stores)
    df_orders = pd.read_csv(ruta_orders)

    # Normalize column names
    df_couriers.columns = [c.strip() for c in df_couriers.columns]
    df_stores.columns = [c.strip() for c in df_stores.columns]
    df_orders.columns = [c.strip() for c in df_orders.columns]

    # Parse time columns
    for col in ['on_time', 'off_time']:
        if col in df_couriers.columns:
            df_couriers[col] = df_couriers[col].apply(to_time_obj)

    for col in ['placement_time', 'preparation_time', 'ready_time', 'expected_drop_off_time']:
        if col in df_orders.columns:
            df_orders[col] = df_orders[col].apply(to_time_obj)

    # Merge store coordinates into orders
    if 'store_id' in df_orders.columns and 'store_id' in df_stores.columns:
        df_orders = df_orders.merge(
            df_stores[['store_id', 'pick_up_lat', 'pick_up_lng']],
            on='store_id',
            how='left'
        )

    # Simulation window
    start_min = 0.0  # Midnight
    end_min = 1439.99  # 23:59:59

    # Build orders dictionary
    orders = {}
    for idx, row in df_orders.iterrows():
        oid = row.get('order_id', idx)
        orders[oid] = {
            'order_id': oid,
            'store_id': row.get('store_id'),
            'pickup_lat': row.get('pick_up_lat'),
            'pickup_lng': row.get('pick_up_lng'),
            'drop_lat': row.get('drop_off_lat'),
            'drop_lng': row.get('drop_off_lng'),
            'placement_min': time_to_min(row.get('placement_time')),
            'preparation_min': time_to_min(row.get('preparation_time')),
            'ready_min': time_to_min(row.get('ready_time')),
            'expected_drop_min': time_to_min(row.get('expected_drop_off_time')),
            'assigned': False
        }

    # Build couriers dictionary
    couriers = {}
    for idx, row in df_couriers.iterrows():
        cid = row.get('courier_id', idx)
        couriers[cid] = {
            'courier_id': cid,
            'vehicle': row.get('vehicle', 'motorcycle'),
            'home_lat': row.get('on_lat'),
            'home_lng': row.get('on_lng'),
            'current_lat': row.get('on_lat'),
            'current_lng': row.get('on_lng'),
            'status': 'idle',
            'available_at': time_to_min(row.get('on_time')) or start_min,
            'on_time_min': time_to_min(row.get('on_time')) or start_min,
            'off_time_min': time_to_min(row.get('off_time')) or end_min,
            'assigned_routes': []
        }

    # Build stores dictionary
    stores = {}
    for idx, row in df_stores.iterrows():
        sid = row.get('store_id', idx)
        stores[sid] = {
            'store_id': sid,
            'pick_up_lat': row.get('pick_up_lat'),
            'pick_up_lng': row.get('pick_up_lng')
        }

    # Instance info
    instance_info = {
        'name': instance_name,
        'n_orders': len(orders),
        'n_couriers': len(couriers),
        'n_stores': len(stores),
        'start_min': start_min,
        'end_min': end_min
    }

    return orders, couriers, stores, instance_info


def initialize_solution_state(couriers):
    """
    Create mutable state dictionary for simulation

    Args:
        couriers: Dictionary of courier data

    Returns:
        Dictionary with couriers state (deep copy)
    """
    state = {'couriers': {}}
    for cid, courier in couriers.items():
        state['couriers'][cid] = {
            'courier_id': cid,
            'vehicle': courier['vehicle'],
            'home_lat': courier['home_lat'],
            'home_lng': courier['home_lng'],
            'current_lat': courier['current_lat'],
            'current_lng': courier['current_lng'],
            'status': 'idle',
            'available_at': courier['available_at'],
            'on_time_min': courier['on_time_min'],
            'off_time_min': courier['off_time_min'],
            'assigned_routes': []
        }
    return state


def group_orders_by_store(order_list, orders):
    """
    Group list of order IDs by their store_id

    Args:
        order_list: List of order IDs
        orders: Dictionary of order data

    Returns:
        Dictionary {store_id: [order_ids]}
    """
    store_groups = {}
    for oid in order_list:
        order = orders[oid]
        store_id = order['store_id']
        if store_id not in store_groups:
            store_groups[store_id] = []
        store_groups[store_id].append(oid)
    return store_groups


def get_available_couriers(solution_state, current_time, couriers):
    """
    Filter couriers that are available at current_time

    Args:
        solution_state: Current simulation state
        current_time: Current simulation time (minutes)
        couriers: Dictionary of courier data

    Returns:
        List of available courier IDs
    """
    available = []
    for cid, c in solution_state['couriers'].items():
        if c['status'] != 'idle':
            continue
        if c['available_at'] is not None and current_time < c['available_at'] - 1e-6:
            continue
        if c['on_time_min'] is not None and current_time < c['on_time_min'] - 1e-6:
            continue
        if c['off_time_min'] is not None and current_time > c['off_time_min'] + 1e-6:
            continue
        available.append(cid)
    return available


def release_finished_couriers(solution_state, current_time):
    """
    Update status of couriers who finished their routes

    Args:
        solution_state: Current simulation state (modified in-place)
        current_time: Current simulation time (minutes)
    """
    for cid, c in solution_state['couriers'].items():
        if c['status'] == 'busy' and c['available_at'] is not None:
            if current_time >= c['available_at'] - 1e-6:
                c['status'] = 'idle'
                c['current_lat'] = c['home_lat']
                c['current_lng'] = c['home_lng']


def save_results(solution, output_path, couriers, execution_time_seconds=None):
    """
    Save solution to Excel file matching notebook format

    Args:
        solution: Solution dictionary with assignments
        output_path: Output file path (.xlsx)
        couriers: Couriers dictionary for summary
        execution_time_seconds: Time taken to execute algorithm (optional)
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build assignments dataframe
    df_assign = pd.DataFrame([
        {
            'time_depart_min': a['time_depart_min'],
            'courier_id': a['courier_id'],
            'store_id': a['store_id'],
            'orders': ','.join(map(str, a['orders'])),
            'depart_store_min': a['depart_store_min'],
            'finish_min': a['finish_min'],
            'total_travel_min': a['total_travel_min']
        }
        for a in solution['assignments']
    ])

    # Build couriers summary dataframe
    resumen_couriers = pd.DataFrame([
        {
            'courier_id': cid,
            'vehicle': c['vehicle'],
            'assigned_routes_count': len(c.get('assigned_routes', [])),
            'next_available_min': c.get('available_at')
        }
        for cid, c in couriers.items()
    ])

    # Calculate number of unique couriers used
    unique_couriers_used = len(set(a['courier_id'] for a in solution['assignments']))

    # Build statistics dataframe
    df_stats = pd.DataFrame([{
        'Tiempo_ejecucion_segundos': execution_time_seconds if execution_time_seconds is not None else 'N/A',
        'Ordenes_totales': solution.get('total_orders', 0),
        'Ordenes_cumplidas': solution.get('orders_covered', 0),
        'Tasa_cobertura': solution.get('coverage_rate', 0),
        'Domiciliarios_utilizados': unique_couriers_used,
        'Rutas_creadas': len(solution['assignments']),
        'Tiempo_viaje_total_min': solution.get('total_travel_time', 0),
        'Tiempo_viaje_promedio_orden': solution.get('avg_travel_per_order', 0)
    }])

    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_assign.to_excel(writer, sheet_name='Asignaciones', index=False)
        resumen_couriers.to_excel(writer, sheet_name='Couriers', index=False)
        df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)

    print(f"Results saved to: {output_path}")

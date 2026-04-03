import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, time as dt_time
from math import radians, sin, cos, sqrt, atan2
from itertools import permutations
import gc

# ============================================
# CONFIGURACIÓN
# ============================================
LOOKAHEAD_PERIODS = 4  # 4 períodos × 2 min = 8 minutos
LOOKAHEAD_MINUTES = LOOKAHEAD_PERIODS * 2  # 8 minutos
MAX_TIME_PER_WINDOW = 120  # 2 minutos = 120 segundos
ROLLING_HORIZON_STEP = 2.0  # minutos
MAX_DISTANCE_KM = 10.0  # Solo couriers dentro de 10 km de la tienda
EARLY_STOP_THRESHOLD = 15.0  # Si encuentra ruta < 15 min, parar búsqueda
MAX_GROUP_SIZE = 2  # Máximo 2 órdenes por grupo

# === PARÁMETROS DE ESTRATEGIA 1 (MAX-COVERAGE) ===
MAX_CANDIDATES_PER_STORE = 10  # Couriers a evaluar por lote por tienda
MAX_BATCHES_PER_STORE = 3      # Máximo de grupos no-solapados por tienda

# ============================================
# FUNCIONES AUXILIARES
# ============================================

speed_map = {'motorcycle': 25.0, 'bicycle': 20.0, 'car': 15.0, 'walking': 5.0}

def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula distancia haversine (sin caché para evitar MemoryError)"""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def haversine_vectorized(lats, lons, store_lat, store_lng):
    """
    Calcula distancias haversine de manera vectorizada (NumPy).
    Mucho más eficiente que un loop Python con caché.
    """
    R = 6371.0
    phi1 = np.radians(lats)
    phi2 = np.radians(store_lat)
    dphi = np.radians(store_lat - lats)
    dlambda = np.radians(store_lng - lons)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def travel_time_min(lat1, lon1, lat2, lon2, vehicle_type):
    """Calcula tiempo de viaje en minutos"""
    if pd.isnull(lat1) or pd.isnull(lat2) or pd.isnull(lon1) or pd.isnull(lon2):
        return 1e6
    sp = speed_map.get(str(vehicle_type).lower(), 15.0)
    dist = haversine_km(lat1, lon1, lat2, lon2)
    return (dist / sp) * 60.0

def time_to_min(t):
    """Convierte time object a minutos"""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if isinstance(t, dt_time):
        return t.hour*60 + t.minute + t.second/60.0
    try:
        tt = pd.to_datetime(t).time()
        return tt.hour*60 + tt.minute + tt.second/60.0
    except:
        return None

def to_time_obj(x):
    """Convierte string a time object"""
    try:
        return pd.to_datetime(x).time()
    except:
        return None

# ============================================
# CARGA DE DATOS
# ============================================

def load_instance(instance_name):
    """Carga una instancia desde las carpetas de datos"""
    carpeta_principal = 'Instancias'
    subcarpeta = instance_name

    ruta_domiciliarios = os.path.join(carpeta_principal, subcarpeta, 'couriers.csv')
    ruta_tiendas = os.path.join(carpeta_principal, subcarpeta, 'stores.csv')
    ruta_ordenes = os.path.join(carpeta_principal, subcarpeta, 'orders.csv')

    df_couriers = pd.read_csv(ruta_domiciliarios)
    df_stores = pd.read_csv(ruta_tiendas)
    df_orders = pd.read_csv(ruta_ordenes)

    # Normalizar nombres
    df_couriers.columns = [c.strip() for c in df_couriers.columns]
    df_stores.columns = [c.strip() for c in df_stores.columns]
    df_orders.columns = [c.strip() for c in df_orders.columns]

    # Parsear tiempos
    for col in ['on_time', 'off_time']:
        if col in df_couriers.columns:
            df_couriers[col] = df_couriers[col].apply(to_time_obj)

    for col in ['placement_time', 'preparation_time', 'ready_time', 'expected_drop_off_time']:
        if col in df_orders.columns:
            df_orders[col] = df_orders[col].apply(to_time_obj)

    # Merge store coords
    if 'store_id' in df_orders.columns and 'store_id' in df_stores.columns:
        df_orders = df_orders.merge(
            df_stores[['store_id', 'pick_up_lat', 'pick_up_lng']],
            on='store_id',
            how='left'
        )

    return df_couriers, df_stores, df_orders, subcarpeta

# ============================================
# PREPARACIÓN DE ESTRUCTURAS
# ============================================

def prepare_orders(df_orders):
    """Convierte DataFrame de órdenes a diccionario"""
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
    return orders

def prepare_couriers(df_couriers, start_min, end_min):
    """Convierte DataFrame de couriers a diccionario"""
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
    return couriers

# ============================================
# FUNCIONES DE OPTIMIZACIÓN
# ============================================

def courier_available(courier, cur_min):
    """Verifica si un courier está disponible en cur_min"""
    if courier['status'] != 'idle':
        return False
    if courier['available_at'] is not None and cur_min < courier['available_at'] - 1e-6:
        return False
    if courier['on_time_min'] is not None and cur_min < courier['on_time_min'] - 1e-6:
        return False
    if courier['off_time_min'] is not None and cur_min > courier['off_time_min'] + 1e-6:
        return False
    return True

def nearest_neighbor_route(order_list, start_lat, start_lng, vehicle):
    """Heurística nearest neighbor para ordenar entregas."""
    if not order_list:
        return []

    remaining = list(order_list)
    route = []
    cur_lat, cur_lng = start_lat, start_lng

    while remaining:
        nearest = min(remaining, key=lambda o: haversine_km(
            cur_lat, cur_lng, o['drop_lat'], o['drop_lng']
        ))
        route.append(nearest)
        cur_lat, cur_lng = nearest['drop_lat'], nearest['drop_lng']
        remaining.remove(nearest)

    return route

def evaluate_route_feasibility(order_sequence, store_lat, store_lng, courier, start_min):
    """
    Evalúa si una secuencia de órdenes es factible y calcula métricas.
    Retorna None si no es factible.
    """
    vehicle = courier['vehicle']

    # Tiempo de viaje a la tienda
    t_to_store = travel_time_min(
        courier['current_lat'], courier['current_lng'],
        store_lat, store_lng, vehicle
    )
    arrive_store_min = start_min + t_to_store

    # Esperar hasta que todas estén listas
    group_ready_min = max([od['ready_min'] for od in order_sequence])
    depart_store_min = max(arrive_store_min, group_ready_min)

    # Simular entregas
    cur_time = depart_store_min
    cur_lat, cur_lng = store_lat, store_lng
    deliveries = []

    for od in order_sequence:
        tt = travel_time_min(cur_lat, cur_lng, od['drop_lat'], od['drop_lng'], vehicle)
        cur_time += tt

        # Verificar deadline
        exp = od.get('expected_drop_min')
        if exp is not None and cur_time > exp + 1e-6:
            return None  # No factible

        deliveries.append({
            'order_id': od['order_id'],
            'drop_time_min': cur_time,
            'drop_lat': od['drop_lat'],
            'drop_lng': od['drop_lng']
        })
        cur_lat, cur_lng = od['drop_lat'], od['drop_lng']

    # Tiempo de regreso
    t_back = travel_time_min(cur_lat, cur_lng, courier['home_lat'], courier['home_lng'], vehicle)
    finish_min = cur_time + t_back
    total_travel = finish_min - start_min

    return {
        'sequence': [d['order_id'] for d in deliveries],
        'deliveries': deliveries,
        'depart_store_min': depart_store_min,
        'finish_min': finish_min,
        'total_travel': total_travel
    }

def best_route_for_courier_and_orders(courier, store_lat, store_lng, order_list, start_min):
    """
    Encuentra la mejor ruta para un courier con un grupo de órdenes.
    Usa heurísticas en lugar de todas las permutaciones.
    """
    n_orders = len(order_list)

    candidates = []

    # Para 1 orden: solo hay 1 permutación
    if n_orders == 1:
        return evaluate_route_feasibility(order_list, store_lat, store_lng, courier, start_min)

    # Para 2+ órdenes: usar heurísticas
    vehicle = courier['vehicle']

    # Heurística 1: Nearest Neighbor
    nn_route = nearest_neighbor_route(order_list, store_lat, store_lng, vehicle)
    result = evaluate_route_feasibility(nn_route, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)
        if result['total_travel'] < EARLY_STOP_THRESHOLD:
            return result

    # Heurística 2: Ordenar por deadline
    sorted_by_deadline = sorted(order_list, key=lambda x: x.get('expected_drop_min', 1e9))
    result = evaluate_route_feasibility(sorted_by_deadline, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)
        if result['total_travel'] < EARLY_STOP_THRESHOLD:
            return result

    # Heurística 3: Ordenar por distancia desde tienda
    sorted_by_distance = sorted(order_list, key=lambda x: haversine_km(
        store_lat, store_lng, x['drop_lat'], x['drop_lng']
    ))
    result = evaluate_route_feasibility(sorted_by_distance, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)

    if not candidates:
        return None

    return min(candidates, key=lambda r: r['total_travel'])

def get_ready_orders(orders, cur_min):
    """Obtiene órdenes que YA fueron colocadas (placement_min <= cur_min)."""
    ready_orders = []
    for o in orders.values():
        if o['assigned']:
            continue
        if o['placement_min'] is not None and o['placement_min'] <= cur_min + 1e-6:
            ready_orders.append(o)
    return ready_orders

def forecast_future_demand(orders, cur_min, lookahead_min, window_size=30):
    """
    Pronóstico de demanda futura combinando tendencia reciente e histórica.
    Retorna dict {store_id: estimated_orders_count}
    """
    if cur_min < 10:
        return {}

    # Ventana reciente (últimos 10 min)
    recent_start = max(0, cur_min - 10)
    recent_orders = [o for o in orders.values()
                     if o['placement_min'] is not None and recent_start <= o['placement_min'] <= cur_min + 1e-6]

    # Ventana histórica (últimos 30 min)
    hist_start = max(0, cur_min - window_size)
    historical_orders = [o for o in orders.values()
                         if o['placement_min'] is not None and hist_start <= o['placement_min'] <= cur_min + 1e-6]

    if not recent_orders and not historical_orders:
        return {}

    hour_of_day = (cur_min // 60) % 24
    is_peak_hour = (11 <= hour_of_day <= 14) or (18 <= hour_of_day <= 21)
    hour_multiplier = 1.3 if is_peak_hour else 0.8

    recent_store_counts = {}
    for o in recent_orders:
        recent_store_counts[o['store_id']] = recent_store_counts.get(o['store_id'], 0) + 1

    hist_store_counts = {}
    for o in historical_orders:
        hist_store_counts[o['store_id']] = hist_store_counts.get(o['store_id'], 0) + 1

    forecast = {}
    all_stores = set(recent_store_counts.keys()) | set(hist_store_counts.keys())

    for store_id in all_stores:
        recent_count = recent_store_counts.get(store_id, 0)
        hist_count = hist_store_counts.get(store_id, 0)

        recent_window = max(1, cur_min - recent_start)
        hist_window = max(1, cur_min - hist_start)

        recent_rate = recent_count / recent_window
        hist_rate = hist_count / hist_window
        combined_rate = (0.7 * recent_rate) + (0.3 * hist_rate)
        adjusted_rate = combined_rate * hour_multiplier
        estimated = adjusted_rate * lookahead_min

        if estimated > 0.5:
            forecast[store_id] = max(1, int(round(estimated)))

    return forecast

def filter_nearby_couriers(couriers_list, store_lat, store_lng, max_distance_km=MAX_DISTANCE_KM):
    """
    Filtra couriers dentro de la distancia máxima usando NumPy vectorizado.
    """
    if not couriers_list:
        return []

    n = len(couriers_list)
    lats = np.empty(n, dtype=np.float64)
    lons = np.empty(n, dtype=np.float64)
    for i, (cid, c) in enumerate(couriers_list):
        lats[i] = c['current_lat'] if c['current_lat'] is not None else 0.0
        lons[i] = c['current_lng'] if c['current_lng'] is not None else 0.0

    distances = haversine_vectorized(lats, lons, store_lat, store_lng)
    mask = distances <= max_distance_km
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return []

    sorted_indices = indices[np.argsort(distances[indices])]
    return [couriers_list[i] for i in sorted_indices]

# ============================================
# ESTRATEGIA 1: MAX-COVERAGE GLOBAL
# ============================================

def generate_window_candidates(available_couriers_list, stores_dict, stores_pending,
                                cur_min, demand_forecast, window_start_time):
    """
    Genera TODOS los candidatos factibles (courier, grupo, tienda) para la ventana actual.

    Diferencia clave vs greedy:
    - El greedy procesa tiendas UNA POR UNA en orden de prioridad
    - Esta función evalúa TODAS las tiendas simultáneamente
    - Cada candidato es una tupla (courier_id, store_id, grupo_de_órdenes, ruta)
    - Genera múltiples grupos no-solapados por tienda (hasta MAX_BATCHES_PER_STORE)

    Retorna lista de candidatos con su score de cobertura.
    """
    candidates = []
    generation_budget = MAX_TIME_PER_WINDOW * 0.70  # Usar 70% del presupuesto

    for sid, pending in stores_pending.items():
        # Verificar presupuesto de tiempo
        if time.time() - window_start_time > generation_budget:
            break

        store_info = stores_dict.get(sid)
        if store_info is None:
            continue

        store_lat = store_info['store_lat']
        store_lng = store_info['store_lng']

        # === HEURÍSTICA WAIT (igual que lookahead) ===
        # Si hay pocas órdenes actuales pero el pronóstico dice que viene más demanda,
        # y el deadline no es urgente, esperar para hacer lotes más grandes.
        forecasted_demand = demand_forecast.get(sid, 0)
        current_count = len(pending)
        if current_count <= 2 and forecasted_demand >= 3:
            min_time_until_deadline = min(
                o.get('expected_drop_min', 1e9) - cur_min for o in pending
            )
            if min_time_until_deadline > 25:
                continue  # Skip: esperar más demanda

        # Ordenar órdenes por urgencia (deadline más próximo primero)
        pending_sorted = sorted(pending, key=lambda x: x.get('expected_drop_min', 1e9))

        # Filtrar couriers cercanos (vectorizado)
        nearby_couriers = filter_nearby_couriers(
            available_couriers_list, store_lat, store_lng, MAX_DISTANCE_KM
        )
        if not nearby_couriers:
            nearby_couriers = available_couriers_list

        # Limitar couriers a evaluar por lote
        couriers_to_eval = nearby_couriers[:MAX_CANDIDATES_PER_STORE]

        # === GENERAR CANDIDATOS PARA MÚLTIPLES LOTES POR TIENDA ===
        # División no-solapada: lote 1 = [0:k], lote 2 = [k:2k], lote 3 = [2k:3k]
        # Esto permite que múltiples couriers sirvan la misma tienda simultáneamente.
        batch_start = 0
        batch_num = 0

        while batch_start < len(pending_sorted) and batch_num < MAX_BATCHES_PER_STORE:
            k = min(MAX_GROUP_SIZE, len(pending_sorted) - batch_start)
            group = pending_sorted[batch_start:batch_start + k]
            group_ids = frozenset(o['order_id'] for o in group)

            # Calcular urgencia del grupo
            min_deadline = min(o.get('expected_drop_min', 1e9) for o in group)
            time_to_deadline = (min_deadline - cur_min) if min_deadline < 1e9 else 1000.0
            # Urgencia: mayor puntaje si el deadline es más próximo
            urgency_score = max(0.0, 100.0 - time_to_deadline) if time_to_deadline < 100.0 else 0.0

            # Evaluar cada courier para este lote
            for cid, c in couriers_to_eval:
                route = best_route_for_courier_and_orders(
                    c, store_lat, store_lng, group, cur_min
                )
                if route is None:
                    continue

                # Score = cobertura (primaria) + urgencia (secundaria) - tiempo viaje (terciaria)
                # Prioridad absoluta: maximizar número de órdenes cubiertas
                score = k * 10000.0 + urgency_score * 10.0 - route['total_travel']

                candidates.append({
                    'courier_id': cid,
                    'store_id': sid,
                    'orders': group,
                    'order_ids': group_ids,
                    'route': route,
                    'n_orders': k,
                    'score': score,
                    'batch_num': batch_num
                })

            batch_start += k
            batch_num += 1

    return candidates


def select_max_coverage(candidates):
    """
    Selección greedy de cobertura máxima (set-cover aproximado).

    Algoritmo:
    1. Ordenar candidatos por score DESC (más órdenes = mejor, más urgente = mejor)
    2. Iterar: seleccionar candidato si no hay conflicto (courier o orden ya usados)
    3. Un conflicto ocurre cuando el courier o alguna orden ya fue asignado

    Propiedad clave vs greedy secuencial:
    - El greedy secuencial da prioridad a tiendas por orden de procesamiento
    - Este algoritmo da prioridad a la CALIDAD del candidato (score global)
    - Tiendas con órdenes urgentes compiten justamente con tiendas con muchas órdenes

    Retorna lista de candidatos seleccionados (sin conflictos).
    """
    # Ordenar por score descendente
    sorted_cands = sorted(candidates, key=lambda c: c['score'], reverse=True)

    selected = []
    used_couriers = set()
    used_orders = set()

    for cand in sorted_cands:
        # ¿El courier ya fue asignado?
        if cand['courier_id'] in used_couriers:
            continue

        # ¿Alguna orden ya fue asignada a otro candidato?
        if cand['order_ids'] & used_orders:  # Intersección no vacía = conflicto
            continue

        # Candidato válido: seleccionar
        selected.append(cand)
        used_couriers.add(cand['courier_id'])
        used_orders.update(cand['order_ids'])

    return selected


# ============================================
# ALGORITMO PRINCIPAL CON MAX-COVERAGE
# ============================================

def optimize_with_maxcover(couriers, orders, stores_df, start_min=0, end_min=1440):
    """
    Algoritmo de cobertura máxima por ventana (Estrategia 1).

    Diferencia fundamental con el greedy secuencial:

    GREEDY SECUENCIAL (corporativo_lookahead.py):
    - Procesa tiendas una por una en orden de prioridad
    - Asigna el mejor courier disponible a cada tienda
    - Tiendas de alta prioridad toman los mejores couriers primero
    - Tiendas de baja prioridad pueden quedar sin couriers

    MAX-COVERAGE GLOBAL (este algoritmo):
    - Genera TODOS los candidatos factibles para TODAS las tiendas simultáneamente
    - Puntúa cada candidato por cobertura y urgencia
    - Selección greedy resuelve conflictos de couriers globalmente
    - El resultado maximiza órdenes cubiertas por ventana (no depende del orden de tiendas)
    """
    assignments = []
    window_times = []

    cur_min = start_min
    window_idx = 0

    print(f"\n{'='*80}")
    print(f"Iniciando optimización con MAX-COVERAGE (Estrategia 1)...")
    print(f"{'='*80}\n")

    # Pre-construir dict de tiendas para O(1) lookup
    stores_dict = {}
    for _, row in stores_df.iterrows():
        sid = row['store_id']
        stores_dict[sid] = {
            'store_lat': float(row['pick_up_lat']),
            'store_lng': float(row['pick_up_lng'])
        }

    while cur_min <= end_min + 1e-6:
        window_start_time = time.time()
        window_idx += 1

        # Liberar couriers que terminaron sus rutas
        for cid, c in couriers.items():
            if c['status'] == 'busy' and c['available_at'] is not None:
                if cur_min >= c['available_at'] - 1e-6:
                    c['status'] = 'idle'
                    c['current_lat'] = c['home_lat']
                    c['current_lng'] = c['home_lng']

        # Órdenes disponibles en esta ventana
        ready_orders = get_ready_orders(orders, cur_min)

        # Pronóstico para heurística de espera
        demand_forecast = forecast_future_demand(orders, cur_min, LOOKAHEAD_MINUTES)

        if window_idx % 50 == 0 or window_idx == 1:
            available_couriers_count = sum(1 for c in couriers.values() if courier_available(c, cur_min))
            forecast_total = sum(demand_forecast.values())
            print(f"[Ventana {window_idx:4d} | Tiempo {cur_min:6.1f} min] "
                  f"Órdenes REALES: {len(ready_orders):5d} | "
                  f"Demanda PRONOSTICADA: {forecast_total:5d} | "
                  f"Couriers disponibles: {available_couriers_count:5d}")

        # Agrupar órdenes por tienda
        stores_pending = {}
        for o in ready_orders:
            stores_pending.setdefault(o['store_id'], []).append(o)

        # Couriers disponibles (calculados UNA vez por ventana)
        available_couriers_list = [
            (cid, c) for cid, c in couriers.items()
            if courier_available(c, cur_min)
        ]

        # Si no hay nada que hacer, avanzar
        if not stores_pending or not available_couriers_list:
            window_elapsed = time.time() - window_start_time
            window_times.append({
                'time_min': cur_min,
                'execution_seconds': window_elapsed,
                'exceeded_limit': False,
                'candidates_generated': 0,
                'assignments_made': 0
            })
            cur_min += ROLLING_HORIZON_STEP
            continue

        # ========================================================
        # ESTRATEGIA 1: GENERACIÓN GLOBAL + SELECCIÓN MAX-COVERAGE
        # ========================================================

        # Paso 1: Generar candidatos de TODAS las tiendas simultáneamente
        candidates = generate_window_candidates(
            available_couriers_list, stores_dict, stores_pending,
            cur_min, demand_forecast, window_start_time
        )

        # Paso 2: Selección greedy de cobertura máxima (set-cover)
        selected = select_max_coverage(candidates)

        # Paso 3: Ejecutar asignaciones seleccionadas
        assignments_this_window = 0
        for sel in selected:
            cid = sel['courier_id']
            sid = sel['store_id']
            group = sel['orders']
            route = sel['route']

            # Verificar que el courier siga disponible (doble seguridad)
            if couriers[cid]['status'] != 'idle':
                continue

            # Marcar órdenes como asignadas
            for o in group:
                orders[o['order_id']]['assigned'] = True

            # Actualizar estado del courier
            couriers[cid]['status'] = 'busy'
            couriers[cid]['available_at'] = route['finish_min']
            couriers[cid]['assigned_routes'].append({
                'time_depart_min': cur_min,
                'store_id': sid,
                'orders': [o['order_id'] for o in group],
                'finish_min': route['finish_min'],
                'total_travel_min': route['total_travel']
            })

            # Registrar asignación
            assignments.append({
                'time_depart_min': cur_min,
                'courier_id': cid,
                'store_id': sid,
                'orders': [o['order_id'] for o in group],
                'depart_store_min': route['depart_store_min'],
                'deliveries': route['deliveries'],
                'finish_min': route['finish_min'],
                'total_travel_min': route['total_travel'],
                'n_orders': sel['n_orders'],
                'score': sel['score']
            })
            assignments_this_window += 1

        # Registrar métricas de la ventana
        window_elapsed = time.time() - window_start_time
        window_times.append({
            'time_min': cur_min,
            'execution_seconds': window_elapsed,
            'exceeded_limit': window_elapsed > MAX_TIME_PER_WINDOW,
            'candidates_generated': len(candidates),
            'assignments_made': assignments_this_window
        })

        cur_min += ROLLING_HORIZON_STEP

    return assignments, window_times

# ============================================
# GUARDAR RESULTADOS
# ============================================

def save_results(assignments, couriers, orders, window_times, output_folder,
                 instance_name, execution_time_total):
    """Guarda resultados en formato Excel"""
    os.makedirs(output_folder, exist_ok=True)

    # DataFrame de asignaciones
    df_assign = pd.DataFrame([
        {
            'time_depart_min': a['time_depart_min'],
            'courier_id': a['courier_id'],
            'store_id': a['store_id'],
            'orders': ",".join(map(str, a['orders'])),
            'orders_count': len(a['orders']),
            'depart_store_min': a['depart_store_min'],
            'finish_min': a['finish_min'],
            'total_travel_min': a['total_travel_min']
        }
        for a in assignments
    ])

    total_orders = len(orders)
    assigned_orders_count = sum(len(a['orders']) for a in assignments)
    coverage_rate = (assigned_orders_count / total_orders * 100) if total_orders > 0 else 0
    avg_travel = df_assign['total_travel_min'].mean() if not df_assign.empty else 0.0

    unique_couriers = set(a['courier_id'] for a in assignments)

    avg_window_time = np.mean([w['execution_seconds'] for w in window_times])
    max_window_time = max([w['execution_seconds'] for w in window_times])
    exceeded_count = sum(1 for w in window_times if w['exceeded_limit'])
    total_candidates = sum(w.get('candidates_generated', 0) for w in window_times)

    df_stats = pd.DataFrame([{
        'Parametros': (f'MaxCoverage | MaxCandidates={MAX_CANDIDATES_PER_STORE} | '
                       f'MaxBatches={MAX_BATCHES_PER_STORE} | MaxGroup={MAX_GROUP_SIZE} | '
                       f'MaxDist={MAX_DISTANCE_KM}km'),
        'Tiempo_ejecucion_total_segundos': execution_time_total,
        'Tiempo_promedio_ventana_segundos': avg_window_time,
        'Tiempo_maximo_ventana_segundos': max_window_time,
        'Ventanas_con_timeout': exceeded_count,
        'Total_ventanas': len(window_times),
        'Total_candidatos_generados': total_candidates,
        'Ordenes_totales': total_orders,
        'Ordenes_cumplidas': assigned_orders_count,
        'Tasa_cobertura': f'{coverage_rate:.2f}%',
        'Domiciliarios_utilizados': len(unique_couriers),
        'Rutas_creadas': len(assignments),
        'Tiempo_viaje_promedio_min': f'{avg_travel:.2f}'
    }])

    df_window_times = pd.DataFrame(window_times)

    out_path = os.path.join(output_folder, f'resultados_{instance_name}_maxcover.xlsx')

    with pd.ExcelWriter(out_path) as writer:
        df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
        df_assign.to_excel(writer, sheet_name='Asignaciones', index=False)
        df_window_times.to_excel(writer, sheet_name='Tiempos_Ventanas', index=False)

        resumen_couriers = pd.DataFrame([
            {
                'courier_id': cid,
                'vehicle': c['vehicle'],
                'assigned_routes_count': len(c['assigned_routes']),
                'next_available_min': c.get('available_at')
            }
            for cid, c in couriers.items() if len(c['assigned_routes']) > 0
        ])
        resumen_couriers.to_excel(writer, sheet_name='Couriers', index=False)

    print(f"\n{'='*80}")
    print(f"RESULTADOS FINALES - {instance_name}")
    print(f"{'='*80}")
    print(f"Total órdenes:           {total_orders:,}")
    print(f"Órdenes asignadas:       {assigned_orders_count:,}")
    print(f"Tasa de cobertura:       {coverage_rate:.2f}%")
    print(f"Rutas creadas:           {len(assignments):,}")
    print(f"Domiciliarios usados:    {len(unique_couriers):,}")
    print(f"Tiempo promedio viaje:   {avg_travel:.2f} min")
    print(f"Tiempo total ejecución:  {execution_time_total:.2f} s ({execution_time_total/60:.2f} min)")
    print(f"Tiempo promedio/ventana: {avg_window_time:.2f} s")
    print(f"Tiempo máximo/ventana:   {max_window_time:.2f} s")
    print(f"Ventanas con timeout:    {exceeded_count}/{len(window_times)}")
    print(f"Candidatos generados:    {total_candidates:,}")
    print(f"\nOK Resultados guardados en: {out_path}")
    print(f"{'='*80}\n")

    return out_path

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        instance_name = sys.argv[1]
    else:
        instance_name = 'Instancia_4'  # Default para prueba inicial

    print(f"\n{'='*80}")
    print(f"MODELO CORPORATIVO - MAX-COVERAGE POR VENTANA (Estrategia 1)")
    print(f"{'='*80}")
    print(f"Instancia:              {instance_name}")
    print(f"Lookahead:              {LOOKAHEAD_MINUTES} min ({LOOKAHEAD_PERIODS} períodos)")
    print(f"Rolling Horizon:        {ROLLING_HORIZON_STEP} min")
    print(f"Max tiempo/ventana:     {MAX_TIME_PER_WINDOW} s")
    print(f"Max distancia courier:  {MAX_DISTANCE_KM} km")
    print(f"Max candidatos/tienda:  {MAX_CANDIDATES_PER_STORE}")
    print(f"Max lotes/tienda:       {MAX_BATCHES_PER_STORE}")
    print(f"Max órdenes/grupo:      {MAX_GROUP_SIZE}")
    print(f"{'='*80}\n")

    print("Cargando datos...")
    start_load = time.time()
    df_couriers, df_stores, df_orders, subcarpeta = load_instance(instance_name)
    print(f"OK Datos cargados en {time.time() - start_load:.2f}s")
    print(f"  - Couriers: {len(df_couriers):,}")
    print(f"  - Stores:   {len(df_stores):,}")
    print(f"  - Orders:   {len(df_orders):,}\n")

    print("Preparando estructuras...")
    start_min = 0
    end_min = 1440
    orders = prepare_orders(df_orders)
    couriers = prepare_couriers(df_couriers, start_min, end_min)

    del df_couriers, df_orders
    gc.collect()
    print(f"OK Estructuras preparadas\n")

    start_opt = time.time()
    assignments, window_times = optimize_with_maxcover(
        couriers, orders, df_stores, start_min, end_min
    )
    execution_time_total = time.time() - start_opt

    output_folder = 'outputs_corporativo_maxcover'
    save_results(
        assignments, couriers, orders, window_times,
        output_folder, instance_name, execution_time_total
    )

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
MAX_COURIERS_PER_STORE = 100  # Límite de couriers a evaluar por tienda (OPTIMIZADO)
MAX_DISTANCE_KM = 10.0  # Solo couriers dentro de 10 km de la tienda (OPTIMIZADO)
EARLY_STOP_THRESHOLD = 15.0  # Si encuentra ruta < 15 min, parar búsqueda (OPTIMIZADO)
MAX_GROUP_SIZE = 2  # Máximo 2 órdenes por grupo (OPTIMIZADO, antes era 3)

# ============================================
# FUNCIONES AUXILIARES CON CACHÉ
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
    """
    Heurística nearest neighbor para ordenar entregas.
    Más rápida que evaluar todas las permutaciones.
    """
    if not order_list:
        return []

    remaining = list(order_list)
    route = []
    cur_lat, cur_lng = start_lat, start_lng

    while remaining:
        # Encontrar la orden más cercana
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
    ✅ OPTIMIZADO: Usa heurísticas en lugar de todas las permutaciones.
    """
    n_orders = len(order_list)
    vehicle = courier['vehicle']

    candidates = []

    # Para 1 orden: solo hay 1 permutación
    if n_orders == 1:
        result = evaluate_route_feasibility(order_list, store_lat, store_lng, courier, start_min)
        return result

    # Para 2+ órdenes: usar heurísticas (más rápido que permutaciones)
    # Heurística 1: Nearest Neighbor
    nn_route = nearest_neighbor_route(order_list, store_lat, store_lng, vehicle)
    result = evaluate_route_feasibility(nn_route, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)
        # ✅ EARLY STOPPING: Si la ruta es muy buena, no buscar más
        if result['total_travel'] < EARLY_STOP_THRESHOLD:
            return result

    # Heurística 2: Ordenar por deadline
    sorted_by_deadline = sorted(order_list, key=lambda x: x.get('expected_drop_min', 1e9))
    result = evaluate_route_feasibility(sorted_by_deadline, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)
        # ✅ EARLY STOPPING
        if result['total_travel'] < EARLY_STOP_THRESHOLD:
            return result

    # Heurística 3: Ordenar por distancia desde tienda
    sorted_by_distance = sorted(order_list, key=lambda x: haversine_km(
        store_lat, store_lng, x['drop_lat'], x['drop_lng']
    ))
    result = evaluate_route_feasibility(sorted_by_distance, store_lat, store_lng, courier, start_min)
    if result:
        candidates.append(result)

    # Si no hay candidatos factibles, retornar None
    if not candidates:
        return None

    # Retornar la mejor de todas las heurísticas
    return min(candidates, key=lambda r: r['total_travel'])

def get_ready_orders(orders, cur_min):
    """
    Obtiene órdenes que YA fueron colocadas (placement_min <= cur_min).
    NO mira hacia el futuro - solo usa información disponible en cur_min.

    Nota: Las órdenes pueden no estar listas aún (ready_min > cur_min),
    pero el courier puede ir a la tienda y esperar hasta que estén listas.
    """
    ready_orders = []

    for o in orders.values():
        if o['assigned']:
            continue

        # Orden ya fue colocada (el sistema tiene conocimiento de ella)
        if o['placement_min'] is not None and o['placement_min'] <= cur_min + 1e-6:
            ready_orders.append(o)

    return ready_orders

def forecast_future_demand(orders, cur_min, lookahead_min, window_size=30):
    """
    PRONÓSTICO MEJORADO de demanda futura con múltiples heurísticas.

    Estrategias combinadas:
    1. Tendencia reciente (últimos 10 min) vs histórico (últimos 30 min)
    2. Patrón por hora del día (hora pico vs hora baja)
    3. Momentum de tienda (tiendas que están recibiendo muchas órdenes recientemente)

    Retorna: dict con predicción por tienda {store_id: estimated_orders_count}
    """
    if cur_min < 10:
        return {}  # No hay suficiente histórico

    # === 1. VENTANA RECIENTE (últimos 10 min) - Mayor peso ===
    recent_start = max(0, cur_min - 10)
    recent_orders = []
    for o in orders.values():
        if o['placement_min'] is not None:
            if recent_start <= o['placement_min'] <= cur_min + 1e-6:
                recent_orders.append(o)

    # === 2. VENTANA HISTÓRICA (últimos 30 min) - Menor peso ===
    hist_start = max(0, cur_min - window_size)
    historical_orders = []
    for o in orders.values():
        if o['placement_min'] is not None:
            if hist_start <= o['placement_min'] <= cur_min + 1e-6:
                historical_orders.append(o)

    if not recent_orders and not historical_orders:
        return {}

    # === 3. PATRÓN POR HORA DEL DÍA ===
    hour_of_day = (cur_min // 60) % 24

    # Horas pico típicas de delivery: 11-14 (almuerzo), 18-21 (cena)
    is_peak_hour = (11 <= hour_of_day <= 14) or (18 <= hour_of_day <= 21)
    hour_multiplier = 1.3 if is_peak_hour else 0.8

    # === 4. CÁLCULO DE TASAS POR TIENDA ===

    # Agrupar órdenes recientes por tienda
    recent_store_counts = {}
    for o in recent_orders:
        recent_store_counts[o['store_id']] = recent_store_counts.get(o['store_id'], 0) + 1

    # Agrupar órdenes históricas por tienda
    hist_store_counts = {}
    for o in historical_orders:
        hist_store_counts[o['store_id']] = hist_store_counts.get(o['store_id'], 0) + 1

    # === 5. FORECAST COMBINADO ===
    forecast = {}
    all_stores = set(recent_store_counts.keys()) | set(hist_store_counts.keys())

    for store_id in all_stores:
        recent_count = recent_store_counts.get(store_id, 0)
        hist_count = hist_store_counts.get(store_id, 0)

        # Tasa reciente (últimos 10 min) - Mayor peso
        recent_window = cur_min - recent_start
        if recent_window < 1:
            recent_window = 1
        recent_rate = recent_count / recent_window

        # Tasa histórica (últimos 30 min) - Menor peso
        hist_window = cur_min - hist_start
        if hist_window < 1:
            hist_window = 1
        hist_rate = hist_count / hist_window

        # Combinar tasas: 70% reciente + 30% histórico
        combined_rate = (0.7 * recent_rate) + (0.3 * hist_rate)

        # Aplicar multiplicador de hora del día
        adjusted_rate = combined_rate * hour_multiplier

        # Proyección futura
        estimated = adjusted_rate * lookahead_min

        # Redondear y asegurar mínimo si hay actividad reciente
        if estimated > 0.5:  # Solo pronosticar si hay probabilidad real
            forecast[store_id] = max(1, int(round(estimated)))

    return forecast

def filter_nearby_couriers(couriers_list, store_lat, store_lng, max_distance_km=MAX_DISTANCE_KM):
    """
    Filtra couriers dentro de la distancia máxima usando NumPy vectorizado.
    Sin caché - calcula todas las distancias en una sola operación numpy.
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

    # Ordenar por distancia
    sorted_indices = indices[np.argsort(distances[indices])]
    return [couriers_list[i] for i in sorted_indices]

# ============================================
# ALGORITMO PRINCIPAL CON LOOKAHEAD
# ============================================

def optimize_with_lookahead(couriers, orders, stores_df, start_min=0, end_min=1440):
    """
    Algoritmo greedy mejorado con PRONÓSTICO (lookahead):

    Estrategia:
    - 2 minutos REVELADOS: órdenes reales disponibles ahora
    - 8 minutos PRONOSTICADOS: estimación de demanda futura
    - Total: 10 minutos de información para tomar decisiones
    - Solo asigna órdenes REALES, pero considera el pronóstico para optimizar
    """
    assignments = []
    window_times = []
    forecast_stats = []  # Registrar calidad del pronóstico

    cur_min = start_min
    window_idx = 0

    print(f"\n{'='*80}")
    print(f"Iniciando optimización con PRONÓSTICO (lookahead)...")
    print(f"{'='*80}\n")

    # Pre-construir dict de tiendas para O(1) lookup (evitar O(n) en DataFrame)
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

        # 1. DATOS REVELADOS (2 minutos actuales): Órdenes que YA están listas
        ready_orders = get_ready_orders(orders, cur_min)

        # 2. PRONÓSTICO (8 minutos futuros): Estimar demanda que viene
        demand_forecast = forecast_future_demand(orders, cur_min, LOOKAHEAD_MINUTES)

        # Registrar pronóstico vs realidad (para análisis posterior)
        forecast_stats.append({
            'time_min': cur_min,
            'ready_orders_count': len(ready_orders),
            'forecast_total': sum(demand_forecast.values()),
            'forecast_by_store': dict(demand_forecast)
        })

        if window_idx % 50 == 0 or window_idx == 1:
            available_couriers_count = sum(1 for c in couriers.values() if courier_available(c, cur_min))
            forecast_total = sum(demand_forecast.values())
            print(f"[Ventana {window_idx:4d} | Tiempo {cur_min:6.1f} min] "
                  f"Órdenes REALES: {len(ready_orders):5d} | "
                  f"Demanda PRONOSTICADA: {forecast_total:5d} | "
                  f"Couriers disponibles: {available_couriers_count:5d}")

        # Agrupar ÓRDENES REALES por tienda
        stores_pending = {}
        for o in ready_orders:
            stores_pending.setdefault(o['store_id'], []).append(o)

        # ✅ HEURÍSTICA ACTIVADA: Priorizar tiendas usando pronóstico
        def store_priority_with_forecast(item):
            store_id, pending_orders = item
            real_count = len(pending_orders)
            forecast_count = demand_forecast.get(store_id, 0)

            # Prioridad = órdenes reales + 40% del pronóstico
            # Priorizar tiendas con alta demanda actual + futura esperada
            return real_count + (0.4 * forecast_count)

        stores_sorted = sorted(
            stores_pending.items(),
            key=store_priority_with_forecast,
            reverse=True  # Mayor prioridad primero
        )

        # Computar couriers disponibles UNA vez por ventana (no dentro del loop)
        available_couriers_list = [
            (cid, c) for cid, c in couriers.items()
            if courier_available(c, cur_min)
        ]

        # Procesar cada tienda (priorizando las que tienen más órdenes)
        for sid, pending in stores_sorted:
            # Verificar timeout de ventana
            elapsed = time.time() - window_start_time
            if elapsed > MAX_TIME_PER_WINDOW:
                if window_idx % 50 == 0:
                    print(f"  Timeout en ventana {window_idx} (tiempo {cur_min:.1f} min)")
                break

            # Ordenar órdenes por expected_drop_min
            pending = sorted(pending, key=lambda x: x.get('expected_drop_min', 1e9))

            # Obtener datos de la tienda (O(1) lookup con dict pre-construido)
            store_info = stores_dict.get(sid)
            if store_info is None:
                continue

            store_lat = store_info['store_lat']
            store_lng = store_info['store_lng']

            # Decisión inteligente de espera con pronóstico
            forecasted_demand = demand_forecast.get(sid, 0)
            current_count = len(pending)

            should_wait = False
            if current_count <= 2 and forecasted_demand >= 3:
                min_time_until_deadline = min(
                    o.get('expected_drop_min', 1e9) - cur_min
                    for o in pending
                )
                if min_time_until_deadline > 25:
                    should_wait = True

            if should_wait:
                if window_idx % 100 == 0:
                    print(f"  >> Tienda {sid}: ESPERANDO (actual:{current_count}, "
                          f"pronostico:+{forecasted_demand})")
                continue

            if not available_couriers_list:
                break

            # Filtrar couriers cercanos UNA vez por tienda (vectorizado con NumPy)
            nearby_couriers = filter_nearby_couriers(
                available_couriers_list, store_lat, store_lng, MAX_DISTANCE_KM
            )

            # Fallback: si no hay couriers cercanos, usar todos
            if not nearby_couriers:
                nearby_couriers = available_couriers_list

            # Limitar cantidad de couriers a evaluar
            couriers_to_eval = nearby_couriers[:MAX_COURIERS_PER_STORE]

            # Asignar órdenes de esta tienda
            while pending:
                # Verificar timeout nuevamente
                elapsed = time.time() - window_start_time
                if elapsed > MAX_TIME_PER_WINDOW:
                    break

                if not couriers_to_eval:
                    break

                # Máximo 2 órdenes por grupo
                max_group = min(MAX_GROUP_SIZE, len(pending))
                best_assignment = None
                best_cid = None
                best_group = None

                # Evaluar diferentes tamaños de grupo
                for k in range(1, max_group + 1):
                    group = pending[:k]

                    for cid, c in couriers_to_eval:
                        route = best_route_for_courier_and_orders(
                            c, store_lat, store_lng, group, cur_min
                        )

                        if route is None:
                            continue

                        if best_assignment is None or route['finish_min'] < best_assignment['finish_min']:
                            best_assignment = route
                            best_cid = cid
                            best_group = group

                            # EARLY STOPPING: Si encontramos una ruta excelente, parar
                            if route['total_travel'] < EARLY_STOP_THRESHOLD:
                                break

                    # Si encontramos una ruta excelente, salir del loop de tamaños de grupo
                    if best_assignment and best_assignment['total_travel'] < EARLY_STOP_THRESHOLD:
                        break

                # Si encontramos una asignación, ejecutarla
                if best_assignment is not None:
                    # Marcar órdenes como asignadas
                    for g in best_group:
                        orders[g['order_id']]['assigned'] = True

                    # Actualizar estado del courier
                    couriers[best_cid]['status'] = 'busy'
                    couriers[best_cid]['available_at'] = best_assignment['finish_min']
                    couriers[best_cid]['assigned_routes'].append({
                        'time_depart_min': cur_min,
                        'store_id': sid,
                        'orders': [g['order_id'] for g in best_group],
                        'finish_min': best_assignment['finish_min'],
                        'total_travel_min': best_assignment['total_travel']
                    })

                    # Registrar asignación CON información de pronóstico
                    assignments.append({
                        'time_depart_min': cur_min,
                        'courier_id': best_cid,
                        'store_id': sid,
                        'orders': [g['order_id'] for g in best_group],
                        'depart_store_min': best_assignment['depart_store_min'],
                        'deliveries': best_assignment['deliveries'],
                        'finish_min': best_assignment['finish_min'],
                        'total_travel_min': best_assignment['total_travel'],
                        'orders_type': 'REAL',
                        'forecasted_demand': demand_forecast.get(sid, 0),
                        'decision_context': f'Real:{len(best_group)}, Forecast:+{demand_forecast.get(sid, 0)}'
                    })

                    # Remover órdenes asignadas de pending
                    pending = [p for p in pending if not orders[p['order_id']]['assigned']]

                    # Remover el courier asignado de las listas disponibles
                    available_couriers_list = [(c_id, c) for c_id, c in available_couriers_list if c_id != best_cid]
                    couriers_to_eval = [(c_id, c) for c_id, c in couriers_to_eval if c_id != best_cid]
                else:
                    # No se pudo asignar, pasar a la siguiente tienda
                    break

        # Registrar tiempo de ejecución de la ventana
        window_elapsed = time.time() - window_start_time
        window_times.append({
            'time_min': cur_min,
            'execution_seconds': window_elapsed,
            'exceeded_limit': window_elapsed > MAX_TIME_PER_WINDOW
        })

        # Avanzar al siguiente período
        cur_min += ROLLING_HORIZON_STEP

    return assignments, window_times, forecast_stats

# ============================================
# GUARDAR RESULTADOS
# ============================================

def save_results(assignments, couriers, orders, window_times, forecast_stats, output_folder,
                 instance_name, execution_time_total):
    """Guarda resultados en formato Excel con información de pronóstico"""
    os.makedirs(output_folder, exist_ok=True)

    # DataFrame de asignaciones (con metadata de pronóstico)
    df_assign = pd.DataFrame([
        {
            'time_depart_min': a['time_depart_min'],
            'courier_id': a['courier_id'],
            'store_id': a['store_id'],
            'orders': ",".join(map(str, a['orders'])),
            'orders_count': len(a['orders']),
            'depart_store_min': a['depart_store_min'],
            'finish_min': a['finish_min'],
            'total_travel_min': a['total_travel_min'],
            'orders_type': a.get('orders_type', 'REAL'),
            'forecasted_demand': a.get('forecasted_demand', 0),
            'decision_context': a.get('decision_context', 'N/A')
        }
        for a in assignments
    ])

    # Estadísticas generales
    total_orders = len(orders)
    assigned_orders_count = sum(len(a['orders']) for a in assignments)
    coverage_rate = (assigned_orders_count / total_orders * 100) if total_orders > 0 else 0
    avg_travel = df_assign['total_travel_min'].mean() if not df_assign.empty else 0.0

    # Couriers utilizados
    unique_couriers = set(a['courier_id'] for a in assignments)

    # Estadísticas de ventanas
    avg_window_time = np.mean([w['execution_seconds'] for w in window_times])
    max_window_time = max([w['execution_seconds'] for w in window_times])
    exceeded_count = sum(1 for w in window_times if w['exceeded_limit'])

    # DataFrame de estadísticas
    df_stats = pd.DataFrame([{
        'Parametros': f'Lookahead={LOOKAHEAD_MINUTES}min, MaxTime={MAX_TIME_PER_WINDOW}s, MaxDist={MAX_DISTANCE_KM}km',
        'Tiempo_ejecucion_total_segundos': execution_time_total,
        'Tiempo_promedio_ventana_segundos': avg_window_time,
        'Tiempo_maximo_ventana_segundos': max_window_time,
        'Ventanas_con_timeout': exceeded_count,
        'Total_ventanas': len(window_times),
        'Ordenes_totales': total_orders,
        'Ordenes_cumplidas': assigned_orders_count,
        'Tasa_cobertura': f'{coverage_rate:.2f}%',
        'Domiciliarios_utilizados': len(unique_couriers),
        'Rutas_creadas': len(assignments),
        'Tiempo_viaje_promedio_min': f'{avg_travel:.2f}'
    }])

    # DataFrame de tiempos por ventana
    df_window_times = pd.DataFrame(window_times)

    # DataFrame de pronósticos (para análisis)
    df_forecast = pd.DataFrame([
        {
            'time_min': f['time_min'],
            'ready_orders': f['ready_orders_count'],
            'forecast_total': f['forecast_total'],
            'forecast_detail': str(f['forecast_by_store']) if f['forecast_by_store'] else 'N/A'
        }
        for f in forecast_stats
    ])

    # Guardar en Excel
    out_path = os.path.join(output_folder, f'resultados_{instance_name}_lookahead.xlsx')

    with pd.ExcelWriter(out_path) as writer:
        df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
        df_assign.to_excel(writer, sheet_name='Asignaciones', index=False)
        df_window_times.to_excel(writer, sheet_name='Tiempos_Ventanas', index=False)
        df_forecast.to_excel(writer, sheet_name='Pronosticos', index=False)

        # Resumen de couriers
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
    print(f"\nOK Resultados guardados en: {out_path}")
    print(f"{'='*80}\n")

    return out_path

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys

    # Obtener nombre de instancia desde argumentos
    if len(sys.argv) > 1:
        instance_name = sys.argv[1]
    else:
        instance_name = 'Instancia_18'  # Default

    print(f"\n{'='*80}")
    print(f"MODELO CORPORATIVO OPTIMIZADO CON LOOKAHEAD")
    print(f"{'='*80}")
    print(f"Instancia:              {instance_name}")
    print(f"Lookahead:              {LOOKAHEAD_MINUTES} min ({LOOKAHEAD_PERIODS} períodos)")
    print(f"Rolling Horizon:        {ROLLING_HORIZON_STEP} min")
    print(f"Max tiempo/ventana:     {MAX_TIME_PER_WINDOW} s")
    print(f"Max distancia courier:  {MAX_DISTANCE_KM} km")
    print(f"Max couriers/tienda:    {MAX_COURIERS_PER_STORE}")
    print(f"{'='*80}\n")

    # Cargar datos
    print("Cargando datos...")
    start_load = time.time()
    df_couriers, df_stores, df_orders, subcarpeta = load_instance(instance_name)
    print(f"OK Datos cargados en {time.time() - start_load:.2f}s")
    print(f"  - Couriers: {len(df_couriers):,}")
    print(f"  - Stores:   {len(df_stores):,}")
    print(f"  - Orders:   {len(df_orders):,}\n")

    # Preparar estructuras
    print("Preparando estructuras...")
    start_min = 0
    end_min = 1440
    orders = prepare_orders(df_orders)
    couriers = prepare_couriers(df_couriers, start_min, end_min)

    # Liberar DataFrames pesados de couriers y órdenes (ya no se necesitan)
    del df_couriers, df_orders
    gc.collect()
    print(f"OK Estructuras preparadas\n")

    # Ejecutar optimización
    start_opt = time.time()

    assignments, window_times, forecast_stats = optimize_with_lookahead(
        couriers, orders, df_stores, start_min, end_min
    )

    execution_time_total = time.time() - start_opt

    # Guardar resultados
    output_folder = 'outputs_corporativo_lookahead'
    save_results(
        assignments, couriers, orders, window_times, forecast_stats,
        output_folder, instance_name, execution_time_total
    )

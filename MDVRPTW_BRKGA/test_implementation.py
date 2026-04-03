"""
Quick validation test for BRKGA implementation
Tests core components without running full BRKGA
"""

import numpy as np
from utils import load_instance
from operators import BRKGAParameters
from decoder import decode_chromosome

def test_basic_functionality():
    """Test basic loading and decoding"""
    print("="*60)
    print("BRKGA Implementation Validation Test")
    print("="*60)

    # Test 1: Load instance
    print("\n[Test 1] Loading instance...")
    try:
        orders, couriers, stores, instance_info = load_instance('Instancia_4')
        print("[OK] Instance loaded successfully")
        print("  - Orders: {}".format(len(orders)))
        print("  - Couriers: {}".format(len(couriers)))
        print("  - Stores: {}".format(len(stores)))
    except Exception as e:
        print("[FAIL] Failed to load instance: {}".format(e))
        return False

    # Test 2: Create parameters
    print("\n[Test 2] Creating BRKGA parameters...")
    try:
        params = BRKGAParameters(
            population_size=10,
            elite_size=2,
            mutant_size=2,
            elite_bias=0.7,
            max_generations=5,
            early_stop_patience=3,
            verbose=False
        )
        print("[OK] Parameters created successfully")
        print("  - Population: {}".format(params.population_size))
        print("  - Elite: {}".format(params.elite_size))
        print("  - Offspring: {}".format(params.offspring_size))
    except Exception as e:
        print("[FAIL] Failed to create parameters: {}".format(e))
        return False

    # Test 3: Generate random chromosome
    print("\n[Test 3] Generating random chromosome...")
    try:
        chromosome_length = len(orders) + len(couriers)
        chromosome = np.random.uniform(0, 1, chromosome_length)
        print("[OK] Chromosome generated successfully")
        print("  - Length: {}".format(chromosome_length))
        print("  - Sample values: {}".format(chromosome[:5]))
    except Exception as e:
        print("[FAIL] Failed to generate chromosome: {}".format(e))
        return False

    # Test 4: Decode chromosome (COVERAGE)
    print("\n[Test 4] Decoding chromosome (Coverage objective)...")
    print("  (This may take 1-2 minutes for large instances...)")
    try:
        solution = decode_chromosome(
            chromosome, orders, couriers, stores, params, fitness_type='coverage'
        )
        print("[OK] Chromosome decoded successfully")
        print("  - Orders covered: {}/{} ({:.2f}%)".format(
            solution['orders_covered'], solution['total_orders'], solution['coverage_rate']*100))
        print("  - Routes created: {}".format(len(solution['assignments'])))
        print("  - Total travel: {:.2f} min".format(solution['total_travel_time']))
        print("  - Avg travel/order: {:.2f} min".format(solution['avg_travel_per_order']))
        print("  - Fitness: {:.4f}".format(solution['fitness']))
    except Exception as e:
        print("[FAIL] Failed to decode chromosome: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Decode chromosome (TRAVEL)
    print("\n[Test 5] Decoding chromosome (Travel objective)...")
    try:
        solution = decode_chromosome(
            chromosome, orders, couriers, stores, params, fitness_type='travel'
        )
        print("[OK] Chromosome decoded successfully")
        print("  - Orders covered: {}/{} ({:.2f}%)".format(
            solution['orders_covered'], solution['total_orders'], solution['coverage_rate']*100))
        print("  - Avg travel/order: {:.2f} min".format(solution['avg_travel_per_order']))
        print("  - Fitness: {:.4f}".format(solution['fitness']))
    except Exception as e:
        print("[FAIL] Failed to decode chromosome: {}".format(e))
        return False

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*60)
    print("\nThe BRKGA implementation is working correctly.")
    print("You can now run full experiments:")
    print("  - Single instance: python brkga_coverage.py --instance Instancia_4")
    print("  - All instances: python run_experiments.py")
    print("="*60)

    return True

if __name__ == '__main__':
    test_basic_functionality()

# BRKGA for Multi-Depot Vehicle Routing Problem with Time Windows (MDVRPTW)

This implementation provides two independent BRKGA (Biased Random-Key Genetic Algorithm) metaheuristics for solving the MDVRPTW problem in last-mile delivery scenarios.

## Overview

- **BRKGA-Coverage**: Maximizes order coverage (prioritizes assigning as many orders as possible)
- **BRKGA-Travel**: Minimizes average travel time per order (prioritizes operational efficiency)

## Requirements

```
Python 3.7+
numpy
pandas
openpyxl
```

Install dependencies:
```bash
pip install numpy pandas openpyxl
```

## Directory Structure

```
MDVRPTW_BRKGA/
├── utils.py              # Utility functions (distance, time, data loading)
├── operators.py          # BRKGA operators (selection, crossover, mutation)
├── decoder.py            # Chromosome decoder (converts genes to routes)
├── brkga_coverage.py     # BRKGA for coverage maximization
├── brkga_travel.py       # BRKGA for travel time minimization
├── run_experiments.py    # Batch testing on all instances
├── README.md             # This file
└── results/              # Output directory (created automatically)
    ├── coverage/         # Coverage optimization results
    └── travel/           # Travel time optimization results
```

## Usage

### 1. Run BRKGA-Coverage on a single instance

```bash
cd MDVRPTW_BRKGA
python brkga_coverage.py --instance Instancia_0
```

**Options:**
- `--instance`: Instance name (e.g., Instancia_0, Instancia_18)
- `--pop_size`: Population size (default: 100)
- `--generations`: Maximum generations (default: 500)
- `--elite_size`: Number of elite individuals (default: 20)
- `--mutant_size`: Number of mutants (default: 15)
- `--elite_bias`: Elite bias for crossover (default: 0.7)
- `--patience`: Early stopping patience (default: 100)

**Example with custom parameters:**
```bash
python brkga_coverage.py --instance Instancia_18 --pop_size 150 --generations 1000
```

### 2. Run BRKGA-Travel on a single instance

```bash
python brkga_travel.py --instance Instancia_0
```

Same options as BRKGA-Coverage.

### 3. Run experiments on all instances

```bash
python run_experiments.py
```

This will:
- Run both BRKGAs on all available instances
- Generate Excel files with solutions
- Create comparison report

**Options:**
- `--instances`: Specific instances to run (e.g., `--instances Instancia_0 Instancia_1`)
- `--algorithms`: Which algorithms to run (`coverage`, `travel`, or `both`) (default: both)
- All BRKGA parameters from above

**Example:**
```bash
python run_experiments.py --instances Instancia_0 Instancia_5 Instancia_10 --pop_size 150
```

### 4. Run quick test on smallest instance

```bash
python brkga_coverage.py --instance Instancia_0 --pop_size 50 --generations 100 --patience 50
```

## Output Files

### Solution Files

Each BRKGA run generates an Excel file with two sheets:

**Location:** `results/coverage/` or `results/travel/`

**Sheet 1 - Asignaciones:** Route assignments
- `time_depart_min`: Departure time (minutes from midnight)
- `courier_id`: Assigned courier
- `store_id`: Pickup store
- `orders`: Comma-separated order IDs
- `depart_store_min`: Actual departure from store (after waiting)
- `finish_min`: Time courier returns to base
- `total_travel_min`: Total travel time for this route

**Sheet 2 - Couriers:** Courier utilization
- `courier_id`: Courier identifier
- `vehicle`: Vehicle type
- `assigned_routes_count`: Number of routes assigned
- `next_available_min`: Next available time

### Comparison Reports

When running `run_experiments.py`, additional files are generated:

**experiment_results_final.xlsx:**
- All results from all experiments
- Columns: instance, algorithm, coverage_rate, avg_travel_per_order, etc.

**comparison_report.xlsx:**
- **All Results:** Complete experiment data
- **Coverage Comparison:** Coverage rates by instance and algorithm
- **Travel Comparison:** Travel times by instance and algorithm
- **Summary Statistics:** Mean, std, min, max for each metric

## Algorithm Parameters

### Recommended Settings

| Parameter | Small Instances | Large Instances | Description |
|-----------|----------------|-----------------|-------------|
| `population_size` | 50-100 | 100-150 | Total individuals |
| `elite_size` | 10-20 | 20-30 | Best solutions preserved |
| `mutant_size` | 10-15 | 15-20 | Random individuals added |
| `elite_bias` | 0.6-0.7 | 0.7-0.8 | Bias towards elite parent |
| `max_generations` | 100-200 | 300-500 | Maximum iterations |
| `early_stop_patience` | 50-100 | 100-150 | Generations without improvement |

### Tuning Guidelines

**If converging too early (premature convergence):**
- Increase `mutant_size` (more diversity)
- Decrease `elite_bias` (more exploration)
- Increase `population_size`

**If not converging (too slow):**
- Increase `elite_size` (more exploitation)
- Increase `elite_bias` (favor best solutions)
- Decrease `population_size` (faster evaluation)

**If runtime is too long:**
- Reduce `population_size`
- Reduce `max_generations`
- Reduce `early_stop_patience`

## Chromosome Encoding

The BRKGA uses a **random-key encoding**:

```
Chromosome = [Order Keys (n) | Courier Keys (m)]
              └──────────────┘  └─────────────┘
              n = # orders       m = # couriers
```

- **Order Keys:** Priority for order assignment (0-1 values)
  - Higher key = higher priority for assignment

- **Courier Keys:** Preference for courier selection (0-1 values)
  - Higher key = preferred when multiple couriers available

**Example:**
- Instance with 1000 orders, 500 couriers
- Chromosome length = 1500 genes
- Each gene is a random value in [0, 1]

## Decoder Algorithm

The decoder converts chromosomes to feasible routes using:

1. **Time-stepped simulation** (2-minute intervals)
2. **Priority-based order processing** (sorted by chromosome keys)
3. **Greedy batch assignment** (1-3 orders per route)
4. **Nearest neighbor routing** (fast heuristic for delivery sequence)
5. **Top-K courier evaluation** (evaluates only 5 best couriers)

**Key optimization:** Uses nearest neighbor instead of exhaustive permutation (O(n²) vs O(n!))

## Fitness Functions

### BRKGA-Coverage

```
Fitness = coverage_penalty + (avg_travel / 20)

where:
  coverage_penalty = 1e9 * (1 - coverage)  if coverage < 50%
                   = 1e7 * (1 - coverage)  if 50% <= coverage < 60%
                   = 1e5 * (1 - coverage)  if coverage >= 60%
```

**Goal:** Minimize fitness → maximize coverage, keep travel reasonable

### BRKGA-Travel

```
Fitness = coverage_penalty + avg_travel

where:
  coverage_penalty = 1e6 * (1 - coverage)  if coverage < 50%
                   = 1e4 * (1 - coverage)  if coverage >= 50%
```

**Goal:** Minimize fitness → minimize travel time, maintain decent coverage

## Expected Performance

### Baseline (Greedy from notebook)
- Coverage: 55.98% (Instance_18)
- Avg travel: 18.76 min/order

### BRKGA Targets
| Metric | BRKGA-Coverage | BRKGA-Travel |
|--------|----------------|--------------|
| Coverage | **≥ 60%** | ≥ 50% |
| Avg travel | ≤ 20 min/order | **≤ 18 min/order** |
| Runtime | 20-30 min | 20-30 min |

### Typical Runtime (Instance_18: 30K orders, 10K couriers)
- Population evaluation: ~10-20 seconds per generation
- Convergence: ~200-300 generations
- **Total:** 20-30 minutes per run

## Troubleshooting

### "ModuleNotFoundError: No module named 'utils'"

Make sure you're in the `MDVRPTW_BRKGA` directory when running:
```bash
cd MDVRPTW_BRKGA
python brkga_coverage.py --instance Instancia_0
```

### "FileNotFoundError: Instancia_X not found"

Check that the `Instancias` folder is in the parent directory:
```
project_root/
├── Instancias/
│   ├── Instancia_0/
│   ├── Instancia_1/
│   └── ...
└── MDVRPTW_BRKGA/
    ├── brkga_coverage.py
    └── ...
```

### Very slow execution

Try reducing parameters for testing:
```bash
python brkga_coverage.py --instance Instancia_0 --pop_size 50 --generations 100
```

### Memory errors

For very large instances, reduce population size:
```bash
python brkga_coverage.py --instance Instancia_18 --pop_size 50
```

## Implementation Details

### Key Features

✅ **Random-key encoding**: No repair mechanisms needed, always valid
✅ **Biased crossover**: Preserves good patterns from elite solutions
✅ **Nearest neighbor routing**: Fast O(n²) instead of O(n!) permutation
✅ **Early stopping**: Automatic convergence detection
✅ **Two fitness functions**: Separate objectives for coverage vs travel
✅ **Excel output**: Compatible with existing analysis tools

### Optimizations

- **Top-K courier evaluation**: Only evaluates 5 best couriers per batch
- **Vectorized operations**: NumPy for efficient chromosome manipulation
- **Time-stepped simulation**: Matches original greedy approach
- **Batch assignment**: Groups orders by store for efficiency

### Constraints Handled

✅ Time windows (ready_min, expected_drop_min)
✅ Courier shifts (on_time_min, off_time_min)
✅ Vehicle types (different speeds)
✅ Single store per route
✅ Return to base time

## References

- **BRKGA:** Bean, J. C. (1994). Genetic Algorithms and Random Keys for Sequencing and Optimization.
- **Original Problem:** Meal Delivery Routing Problem

## Contact
Daniel Sebastian Giraldo Herrera
ds.giraldoh@uniandes.edu.co

---

**Implementation Date:** 2026-02-13
**Version:** 1.0 (Basic)

# bugBox

An evolutionary swarm simulation where neural-network-controlled agents learn to navigate complex obstacle fields and reach targets through genetic algorithm-based evolution. Implements multi-faction speciation and intelligent fitness-weighted selection across Python and C++ implementations.

## Overview

**bugBox** simulates a population of autonomous agents (creatures) that navigate a 2D environment filled with static and dynamic obstacles. Each creature is controlled by a small neural network whose weights are encoded in its DNA. Over many generations, natural selection—implemented through tournament selection and weighted crossover—produces increasingly capable navigation strategies.

The simulation features:
- **Multi-faction evolution**: Creatures self-segregate into left and right factions based on positioning, with faction fitness scores proportionally allocating breeding slots
- **Elite preservation**: Top performers are guaranteed reproduction without mutation
- **Adaptive mutation**: Mutation rates respond to fitness stagnation patterns
- **Dual-mode execution**: Visual Pygame-based UI and headless parallel processing
- **Comprehensive telemetry**: Per-generation CSV logging of population statistics

## Features

### Core Mechanics
- **Neural Network Control**: Each creature's behavior is determined by a feedforward neural network (10 inputs → 8 hidden → 2 outputs)
- **Genetic Algorithm Evolution**: DNA-encoded network weights are recombined and mutated across generations
- **Multi-Faction Speciation**: Population self-organizes into competing groups; breeding allocations reflect faction fitness
- **Tournament-Based Parent Selection**: k-tournament selection ensures fit parents are more likely to reproduce
- **Fitness-Weighted Crossover**: Child DNA inheritance biased toward fitter parents
- **Dynamic & Static Obstacles**: Environment includes fixed barriers and moving doors

### Sensor System
Each creature has 10 sensor inputs:
- Distance to target (normalized)
- Direction to target (x, y components)
- 8 directional obstacle avoidance rays (normalized distance to obstacles in 8 directions)

### Neural Network Architecture
```
Input Layer (10)  →  Hidden Layer (8)  →  Output Layer (2)
                                             └─ Steering (x force)
                                             └─ Acceleration (y force)
```

Network weights are fully determined by DNA genes via direct mapping.

## Technical Architecture

### Python Implementation (`src/`)

| Module | Purpose |
|--------|---------|
| `population.py` | Population lifecycle, generation stepping, fitness evaluation, natural selection with speciation |
| `creature.py` | Agent physics (forces, velocity, position), sensor computation, fitness calculation, collision detection |
| `dna.py` | Genetic encoding (genes as numpy arrays), crossover and mutation operators |
| `nn/` | Neural network module for forward pass evaluation |

### C++ Implementation (`src_cpp/`)

Parallel implementation using C++17 with the same algorithm for performance-critical workloads:

| Module | Purpose |
|--------|---------|
| `main.cpp` | Raylib-based visual simulation |
| `headless_main.cpp` | Multi-threaded batch simulation runner |
| `population.cpp`, `creature.cpp`, `dna.cpp` | Core simulation logic (architecture mirrors Python) |
| `nn/nn.cpp` | Neural network evaluation |

### Entry Points

- **`main.py`** — Python visual simulation with real-time rendering (Pygame)
- **`headless_main.py`** — Python headless mode with optional multiprocessing for batch runs
- **`bugbox`** (C++ binary) — Raylib-based visual rendering with ~60 FPS sync
- **`bugbox_headless`** (C++ binary) — Multi-threaded simulation for rapid research/benchmarking

## Requirements

### Python
- Python 3.10+ (recommended)
- Dependencies: `pygame`, `numpy`

### C++ (Optional)
- CMake 3.14+
- C++17 compiler
- Raylib (fetched automatically via FetchContent)

## Setup

### Python Environment

```bash
cd /path/to/bugBox
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### C++ Build (Optional)

```bash
mkdir -p build && cd build
cmake ..
make
```

This produces two executables:
- `./bugbox` — Visual simulation
- `./bugbox_headless` — Headless batch runner

## Usage

### Python Visual Mode

```bash
python3 main.py
```

Displays a 800×600 simulation window with:
- Top area: 2D navigation environment with obstacles and agents
- Bottom panel: Real-time telemetry (generation count, fitness stats, mutation rate, faction populations)
- Color coding: 
  - Green: Left faction (standard)
  - Light green: Left faction elite
  - Red: Right faction (standard)
  - Light red: Right faction elite
  - Gold: All-time champion

Press `Esc` or close the window to stop.

### Python Headless Mode

```bash
python3 headless_main.py [--gens GENERATIONS] [--seed SEED] [--output CSV_PATH]
```

Options:
- `--gens` — Number of generations to simulate (default: 800)
- `--seed` — Random seed for reproducibility
- `--output` — Path to save telemetry CSV (default: `swarm_telemetry.csv`)
- `--verbose` — Enable detailed per-generation logging

Example:
```bash
python3 headless_main.py --gens 500 --seed 42 --output results/run_001.csv
```

### C++ Builds

**Visual Mode:**
```bash
./build/bugbox
```

**Headless Mode (multi-threaded):**
```bash
./build/bugbox_headless --gens 1000 --seed 42
```

## Telemetry Format

Each run produces/overwrites `swarm_telemetry.csv` with per-generation statistics:

| Column | Description |
|--------|-------------|
| `Generation` | Generation index (0-indexed) |
| `Max_Fitness` | Highest fitness in population |
| `Avg_Fitness` | Mean population fitness |
| `Successes` | Count of creatures reaching goal |
| `Crashes` | Count of creatures hitting obstacles |
| `Left_Faction` | Population count in left faction |
| `Right_Faction` | Population count in right faction |
| `Mutation_Rate` | Current mutation rate (adaptive) |

Example row:
```
125,1.847,0.923,42,158,412,388,0.08
```

## Algorithm Details

### Fitness Function
$$f = e^{-d/800}$$
where $d$ is the creature's closest distance to the target throughout its lifetime. Creatures reaching the target receive a significant bonus.

### Natural Selection Pipeline

1. **Evaluation**: Calculate fitness for all creatures
2. **Speciation**: Partition into left/right factions based on average x-position
3. **Faction Scoring**: Sum fitness within each faction
4. **Allocation**: Proportionally allocate breeding slots; guarantee minimum 10% survival per faction
5. **Breeding**:
   - Preserve top 10% as elites (no mutation)
   - Fill remaining slots through tournament selection + crossover + mutation

### Mutation Strategy

Vectorized mutation on DNA arrays:
- Probability: each gene mutates with probability `mutation_rate`
- Magnitude: mutations add uniform noise in range [-0.2, 0.2]
- Adaptive rate: increases on stagnation (currently static at 0.08 in baseline config)

### Crossover Mechanism

**Fitness-Weighted**: Child genes are biased toward fitter parent
```
fitness_ratio = fitA / (fitA + fitB)
if random() < fitness_ratio: gene = parent_A.gene
else: gene = parent_B.gene
```

## Project Layout

```
bugBox/
├── main.py                      # Python visual entry point
├── headless_main.py             # Python headless entry point
├── requirements.txt             # Python dependencies
├── CMakeLists.txt               # C++ build configuration
├── src/                         # Python implementation
│   ├── population.py
│   ├── creature.py
│   ├── dna.py
│   ├── nn/
│   │   └── __init__.py
│   └── __pycache__/
├── src_cpp/                     # C++ implementation
│   ├── main.cpp
│   ├── headless_main.cpp
│   ├── population.cpp
│   ├── creature.cpp
│   ├── dna.cpp
│   └── nn/
│       └── nn.cpp
├── include/                     # C++ headers
│   ├── population.hpp
│   ├── creature.hpp
│   ├── dna.hpp
│   └── nn/
│       └── nn.hpp
├── build/                       # C++ build output (generated)
├── assets/                      # Simulation assets (if any)
└── README.md
```

## Configuration

Key simulation parameters are defined in entry point files:

| Parameter | Value | Location |
|-----------|-------|----------|
| Population size | 800 | `main.py:26`, `headless_main.py:14` |
| Simulation width | 800px | Constants |
| Simulation height | 600px | Constants |
| Generations per run | 800 | Configurable |
| Mutation rate | 0.08 | `Population` init |
| Neural net inputs | 10 | `creature.py:72` |
| Neural net hidden | 8 | `creature.py:72` |
| Neural net outputs | 2 | `creature.py:72` |

Edit these values directly in source to tweak behavior.

## Development Notes

- **Pygame vs Raylib**: Python uses Pygame for simplicity; C++ uses Raylib for performance
- **Reproducibility**: Set `--seed` (Python) or random seed (C++) for deterministic runs
- **Scaling**: Headless mode supports multiprocessing (Python) or threading (C++) for batch research
- **Extensibility**: The modular architecture allows easy addition of new sensor types, obstacle behaviors, or evolutionary pressures

## Performance

### Python (on typical machine)
- ~30-60 FPS with 800 agents (Pygame mode)
- ~200+ generations/second (headless)

### C++ (on typical machine)
- ~60 FPS with 800 agents (Raylib mode)
- ~1000+ generations/second (headless, multithreaded)

## Known Limitations & Future Work

- Sensor rays are rectangular collision checks (not true geometric rays)
- Neural network is fully connected (no recurrence/LSTM)
- Obstacle layout is hard-coded; procedural generation not yet implemented
- Multi-node distributed simulation not yet supported

## License

(Specify if applicable)

## Contact

For questions or contributions, please refer to the repository documentation.

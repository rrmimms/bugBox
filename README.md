# bugBox

`bugBox` is a Python + Pygame evolutionary simulation where a swarm of agents learns to navigate obstacle layouts and reach a target over generations.

## Features

- Genetic algorithm-based population evolution
- Dynamic + static obstacle layout in the visual simulation
- Adaptive mutation behavior based on fitness stagnation
- Generation-level CSV telemetry logging

## Requirements

- Python 3.10+ (recommended)
- `pip`

Dependencies are listed in `requirements.txt`.

## Setup

```bash
cd /home/robert/personalProjects/bugBox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Telemetry Output

Each run creates/resets `swarm_telemetry.csv` and appends one row per generation.

Header columns:

- `Generation`
- `Max_Fitness`
- `Avg_Fitness`
- `Successes`
- `Crashes`
- `Left_Faction`
- `Right_Faction`
- `Mutation_Rate`

## Project Layout

- `main.py` — primary visual simulation loop
- `headless_main.py` — alternate/headless entry point
- `src/population.py` — population lifecycle and evolution
- `src/creature.py` — agent behavior/state
- `src/dna.py` — genetic encoding

## Notes

- Close the Pygame window to stop the simulation.
- Telemetry file is overwritten at each new run by design.

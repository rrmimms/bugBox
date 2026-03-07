import sys
import time
import math
import csv
import random
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
# Try to import Population, path might need adjustment or ensure we run from root
from src.population import Population

WIDTH, HEIGHT = 800, 600
POP_SIZE = 800
GEN_TTL = 800
TELEMETRY_UI_ROWS = 10

class SimpleRect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def collidepoint(self, x, y):
        return (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h)


def build_obstacles():
    static_obstacles = [
        SimpleRect(100, 400, 200, 80),
        SimpleRect(500, 400, 200, 80),
        SimpleRect(250, 250, 300, 80),
        SimpleRect(50, 100, 200, 50),
        SimpleRect(550, 100, 200, 50),
    ]
    moving_door_one = SimpleRect(300, 100, 20, 100)
    moving_door_two = SimpleRect(475, 100, 20, 100)
    current_obstacles = static_obstacles + [moving_door_one, moving_door_two]
    return moving_door_one, moving_door_two, current_obstacles


def run_simulation(max_gens=50, telemetry_path=None, seed=None, verbose=True, run_id=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    target_pos = [WIDTH // 2, 50]
    start_pos = [WIDTH // 2, HEIGHT - 50]
    moving_door_one, moving_door_two, current_obstacles = build_obstacles()

    pop = Population(size=POP_SIZE, mutation_rate=.02, start_pos=start_pos, target_pos=target_pos, dna_length=GEN_TTL)

    frame_count = 0
    gen = 1

    best_fitness_ever = 0.0
    best_path_ever = []
    stagnation = 0
    telemetry_history = []
    last_successes = 0
    last_crashes = 0
    last_avg_fit = 0.0
    last_min_distance = float("inf")
    last_avg_life = 0.0
    last_avg_speed = 0.0
    last_dna_diversity = 0.0
    last_success_rate = 0.0
    last_best_finish_time = None
    last_avg_finish_time = None

    peak_successes = 0
    peak_success_rate = 0.0
    first_success_gen = None
    generations_with_success = 0
    fastest_finish_time = None

    if telemetry_path:
        with open(telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Generation",
                "Max_Fitness",
                "Avg_Fitness",
                "Successes",
                "Success_Rate",
                "Crashes",
                "Left_Faction",
                "Right_Faction",
                "Mutation_Rate",
                "Min_Distance",
                "Avg_Lifetime",
                "Avg_Speed",
                "DNA_Diversity",
                "Best_Finish_Time",
                "Avg_Finish_Time",
            ])

    start_time = time.time()

    if verbose:
        print(f"Starting headless simulation for {max_gens} generations...")

    while gen <= max_gens:
        door_y = 100 + math.sin(frame_count * 0.05) * 40
        moving_door_one.y = int(door_y)
        moving_door_two.y = int(door_y)

        pop.update(frame_count, WIDTH, HEIGHT, current_obstacles)
        frame_count += 1

        if all(c.crashed or c.reached_goal for c in pop.creatures):
            frame_count = GEN_TTL

        if frame_count >= GEN_TTL:
            pop.evaluate_fitness()
            successes = sum(1 for c in pop.creatures if c.reached_goal)
            crashes = sum(1 for c in pop.creatures if c.crashed)
            success_rate = (successes / pop.size) if pop.size else 0.0
            fitness_total = 0.0
            max_fit = float("-inf")
            for creature in pop.creatures:
                fitness = creature.fitness
                fitness_total += fitness
                if fitness > max_fit:
                    max_fit = fitness
            avg_fit = fitness_total / pop.size if pop.size else 0.0
            left_count = sum(1 for c in pop.creatures if getattr(c, 'avg_x', 400) < 400)
            right_count = pop.size - left_count

            min_distance = min(c.closest_dist for c in pop.creatures) if pop.creatures else float("inf")
            avg_life = float(np.mean([c.lifetime for c in pop.creatures])) if pop.creatures else 0.0
            avg_speed = (
                float(np.mean([math.hypot(float(c.vel[0]), float(c.vel[1])) for c in pop.creatures]))
                if pop.creatures
                else 0.0
            )
            all_dna = np.array([c.brain.get_dna() for c in pop.creatures], dtype=np.float64) if pop.creatures else np.array([])
            dna_diversity = float(np.var(all_dna)) if all_dna.size > 0 else 0.0

            successful_finish_times = [c.finish_time for c in pop.creatures if c.reached_goal]
            best_finish_time = min(successful_finish_times) if successful_finish_times else None
            avg_finish_time = float(np.mean(successful_finish_times)) if successful_finish_times else None

            if successes > peak_successes:
                peak_successes = successes
            if success_rate > peak_success_rate:
                peak_success_rate = success_rate
            if successes > 0:
                generations_with_success += 1
                if first_success_gen is None:
                    first_success_gen = gen
            if best_finish_time is not None and (fastest_finish_time is None or best_finish_time < fastest_finish_time):
                fastest_finish_time = best_finish_time

            telemetry_history.append((
                gen,
                max_fit,
                avg_fit,
                successes,
                success_rate,
                crashes,
                left_count,
                right_count,
                pop.mutation_rate,
                min_distance,
                avg_life,
                avg_speed,
                dna_diversity,
                best_finish_time,
                avg_finish_time,
            ))
            if len(telemetry_history) > TELEMETRY_UI_ROWS:
                telemetry_history.pop(0)

            if telemetry_path:
                with open(telemetry_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        gen,
                        round(max_fit, 2),
                        round(avg_fit, 2),
                        successes,
                        round(success_rate, 4),
                        crashes,
                        left_count,
                        right_count,
                        pop.mutation_rate,
                        round(min_distance, 3),
                        round(avg_life, 2),
                        round(avg_speed, 4),
                        round(dna_diversity, 6),
                        best_finish_time if best_finish_time is not None else "",
                        round(avg_finish_time, 2) if avg_finish_time is not None else "",
                    ])

            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                best_bug = max(pop.creatures, key=lambda c: c.fitness)
                best_path_ever = best_bug.path_history.copy()
                stagnation = 0
                pop.mutation_rate = 0.01
            else:
                stagnation += 1

            if stagnation == 5:
                if verbose:
                    print("*** Stagnation detected. Spiking mutation! ***")
                pop.mutation_rate = 0.03
            elif stagnation > 7:
                if verbose:
                    print("*** Cooling down mutation to stabilize swarm. ***")
                pop.mutation_rate = 0.01
                stagnation = 0

            if verbose:
                print(f"--- Generation {gen} ---")
                print(
                    f"Success: {successes}/{pop.size} ({success_rate:.2%}) | "
                    f"Crashed: {crashes}/{pop.size}"
                )
                print(f"Max Fit: {max_fit:.5f} | Avg Fit: {avg_fit:.5f}\n")
                print(
                    f"MinDist: {min_distance:.3f} | AvgLife: {avg_life:.2f} | "
                    f"AvgSpeed: {avg_speed:.4f} | DNA Var: {dna_diversity:.6f}\n"
                )
                if best_finish_time is not None:
                    print(
                        f"FinishTime(best/avg): {best_finish_time}/{avg_finish_time:.2f} ticks | "
                        f"PeakSuccess: {peak_successes}/{pop.size} ({peak_success_rate:.2%})\n"
                    )
                else:
                    print(
                        f"FinishTime(best/avg): n/a | "
                        f"PeakSuccess: {peak_successes}/{pop.size} ({peak_success_rate:.2%})\n"
                    )

            pop.natural_selection()
            frame_count = 0
            gen += 1

            if verbose and gen <= max_gens:
                print(f"Generation {gen} has begun...")

            last_successes = successes
            last_crashes = crashes
            last_avg_fit = avg_fit
            last_min_distance = min_distance
            last_avg_life = avg_life
            last_avg_speed = avg_speed
            last_dna_diversity = dna_diversity
            last_success_rate = success_rate
            last_best_finish_time = best_finish_time
            last_avg_finish_time = avg_finish_time

    elapsed = time.time() - start_time
    return {
        "elapsed": elapsed,
        "best_fitness": best_fitness_ever,
        "best_path_points": len(best_path_ever),
        "last_successes": last_successes,
        "last_crashes": last_crashes,
        "last_avg_fit": last_avg_fit,
        "last_min_distance": last_min_distance,
        "last_avg_life": last_avg_life,
        "last_avg_speed": last_avg_speed,
        "last_dna_diversity": last_dna_diversity,
        "last_success_rate": last_success_rate,
        "last_best_finish_time": last_best_finish_time,
        "last_avg_finish_time": last_avg_finish_time,
        "peak_successes": peak_successes,
        "peak_success_rate": peak_success_rate,
        "first_success_gen": first_success_gen,
        "generations_with_success": generations_with_success,
        "fastest_finish_time": fastest_finish_time,
        "max_gens": max_gens,
        "seed": seed,
        "telemetry_path": telemetry_path,
        "run_id": run_id,
        "pid": os.getpid(),
    }


def _run_worker(task):
    return run_simulation(
        max_gens=task["max_gens"],
        telemetry_path=task["telemetry_path"],
        seed=task["seed"],
        verbose=False,
        run_id=task["run_id"],
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run bugBox headless simulation")
    parser.add_argument("max_gens", nargs="?", type=int, default=50, help="Number of generations per run")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--runs", type=int, default=1, help="Total independent runs to execute")
    parser.add_argument("--benchmark", action="store_true", help="Compare 1 worker vs N workers for the same number of runs")
    parser.add_argument(
        "--benchmark-mode",
        choices=["single", "multi", "both"],
        default="both",
        help="Benchmark single-worker only, multi-worker only, or both",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (offset by run index)")
    parser.add_argument("--telemetry", action="store_true", help="Write per-run telemetry CSV files")
    return parser.parse_args(argv)


def run_parallel(max_gens, runs, workers, base_seed=None, write_telemetry=False, log_progress=True, label="parallel"):
    workers = max(1, workers)
    runs = max(1, runs)
    worker_count = min(workers, runs)
    tasks = []
    for run_index in range(runs):
        seed = None if base_seed is None else base_seed + run_index
        telemetry_path = None
        if write_telemetry:
            telemetry_path = f"swarm_telemetry_run_{run_index + 1}.csv"
        tasks.append({
            "max_gens": max_gens,
            "seed": seed,
            "telemetry_path": telemetry_path,
            "run_id": run_index + 1,
        })

    if log_progress:
        print(
            f"[{label}] Launching {runs} run(s) across {worker_count} worker(s) "
            f"for {max_gens} generation(s) per run..."
        )

    start_time = time.time()
    results = []
    completed_count = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_worker, task) for task in tasks]
        for completed in as_completed(futures):
            result = completed.result()
            results.append(result)
            completed_count += 1
            if log_progress:
                wall_elapsed = time.time() - start_time
                throughput = completed_count / wall_elapsed if wall_elapsed > 0 else 0.0
                print(
                    f"[{label}] Run {completed_count}/{runs} finished "
                    f"(run_id={result['run_id']}, pid={result['pid']}, seed={result['seed']}) | "
                    f"run_time={result['elapsed']:.3f}s best_fit={result['best_fitness']:.5f} "
                    f"throughput={throughput:.2f} runs/s"
                )
    elapsed = time.time() - start_time

    best_result = max(results, key=lambda item: item["best_fitness"]) if results else None
    avg_run_time = (sum(item["elapsed"] for item in results) / len(results)) if results else 0.0
    min_run_time = min((item["elapsed"] for item in results), default=0.0)
    max_run_time = max((item["elapsed"] for item in results), default=0.0)
    aggregate_throughput = (runs / elapsed) if elapsed > 0 else 0.0

    if log_progress:
        print(
            f"[{label}] All runs complete | wall_time={elapsed:.3f}s "
            f"aggregate_throughput={aggregate_throughput:.2f} runs/s "
            f"avg_run={avg_run_time:.3f}s min_run={min_run_time:.3f}s max_run={max_run_time:.3f}s"
        )

    return {
        "elapsed": elapsed,
        "results": results,
        "best_result": best_result,
        "runs": runs,
        "workers": worker_count,
        "aggregate_throughput": aggregate_throughput,
        "avg_run_time": avg_run_time,
        "min_run_time": min_run_time,
        "max_run_time": max_run_time,
    }

def main():
    args = parse_args(sys.argv[1:])

    if args.benchmark:
        print(
            f"Benchmarking mode={args.benchmark_mode} | "
            f"{args.runs} run(s) x {args.max_gens} generation(s) each..."
        )

        single = None
        multi = None

        if args.benchmark_mode in ("single", "both"):
            single = run_parallel(
                max_gens=args.max_gens,
                runs=args.runs,
                workers=1,
                base_seed=args.seed,
                write_telemetry=False,
                log_progress=True,
                label="single",
            )
            print(f"Single-worker time: {single['elapsed']:.4f}s")
            print(f"Single-worker throughput: {single['aggregate_throughput']:.2f} runs/s")
            if single["best_result"] is not None:
                best_single = single["best_result"]
                print(
                    f"Single best solve telemetry: peak_success={best_single['peak_successes']}/{POP_SIZE} "
                    f"({best_single['peak_success_rate']:.2%}), first_success_gen={best_single['first_success_gen']}, "
                    f"gens_with_success={best_single['generations_with_success']}, "
                    f"fastest_finish={best_single['fastest_finish_time']}"
                )

        if args.benchmark_mode in ("multi", "both"):
            multi = run_parallel(
                max_gens=args.max_gens,
                runs=args.runs,
                workers=args.workers,
                base_seed=args.seed,
                write_telemetry=False,
                log_progress=True,
                label="multi",
            )
            print(f"{multi['workers']}-worker time: {multi['elapsed']:.4f}s")
            print(f"{multi['workers']}-worker throughput: {multi['aggregate_throughput']:.2f} runs/s")
            if multi["best_result"] is not None:
                print(f"Best fitness observed: {multi['best_result']['best_fitness']:.5f}")
                best_multi = multi["best_result"]
                print(
                    f"Multi best solve telemetry: peak_success={best_multi['peak_successes']}/{POP_SIZE} "
                    f"({best_multi['peak_success_rate']:.2%}), first_success_gen={best_multi['first_success_gen']}, "
                    f"gens_with_success={best_multi['generations_with_success']}, "
                    f"fastest_finish={best_multi['fastest_finish_time']}"
                )

        if single is not None and multi is not None:
            speedup = single["elapsed"] / multi["elapsed"] if multi["elapsed"] > 0 else 0.0
            efficiency = speedup / multi["workers"] if multi["workers"] > 0 else 0.0
            print(f"Speedup: {speedup:.2f}x")
            print(f"Parallel efficiency: {efficiency:.2%}")
        return

    if args.runs == 1 and args.workers == 1:
        telemetry_path = "swarm_telemetry.csv" if args.telemetry else None
        result = run_simulation(
            max_gens=args.max_gens,
            telemetry_path=telemetry_path,
            seed=args.seed,
            verbose=True,
        )
        print(f"\nSimulation finished in {result['elapsed']:.4f} seconds.")
        if result["best_path_points"]:
            print(f"Best path points recorded: {result['best_path_points']}")
        print(
            f"Solve telemetry: peak_success={result['peak_successes']}/{POP_SIZE} "
            f"({result['peak_success_rate']:.2%}), first_success_gen={result['first_success_gen']}, "
            f"gens_with_success={result['generations_with_success']}, "
            f"fastest_finish={result['fastest_finish_time']}"
        )
        if telemetry_path:
            print(f"Telemetry written to {telemetry_path}")
        return

    summary = run_parallel(
        max_gens=args.max_gens,
        runs=args.runs,
        workers=args.workers,
        base_seed=args.seed,
        write_telemetry=args.telemetry,
        log_progress=True,
        label="parallel",
    )
    print(
        f"Completed {summary['runs']} run(s) in {summary['elapsed']:.4f}s "
        f"with {summary['workers']} worker(s)."
    )
    print(
        f"Throughput: {summary['aggregate_throughput']:.2f} runs/s | "
        f"avg/min/max run time: {summary['avg_run_time']:.3f}s/"
        f"{summary['min_run_time']:.3f}s/{summary['max_run_time']:.3f}s"
    )
    if summary["best_result"] is not None:
        best = summary["best_result"]
        print(
            f"Best run: fitness={best['best_fitness']:.5f}, "
            f"avg_fit(last_gen)={best['last_avg_fit']:.5f}, "
            f"successes(last_gen)={best['last_successes']} ({best['last_success_rate']:.2%}), "
            f"run_id={best['run_id']}, pid={best['pid']}, seed={best['seed']}"
        )
        print(
            f"Best run extras: min_dist={best['last_min_distance']:.3f}, "
            f"avg_life={best['last_avg_life']:.2f}, "
            f"avg_speed={best['last_avg_speed']:.4f}, "
            f"dna_var={best['last_dna_diversity']:.6f}"
        )
        print(
            f"Best run solve telemetry: peak_success={best['peak_successes']}/{POP_SIZE} "
            f"({best['peak_success_rate']:.2%}), first_success_gen={best['first_success_gen']}, "
            f"gens_with_success={best['generations_with_success']}, "
            f"fastest_finish={best['fastest_finish_time']}"
        )
    if args.telemetry:
        print("Telemetry written to swarm_telemetry_run_<n>.csv files")

if __name__ == "__main__":
    main()

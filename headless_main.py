import sys
import time
import math
import csv
# Try to import Population, path might need adjustment or ensure we run from root
from src.population import Population

WIDTH, HEIGHT = 800, 600
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

def main():
    target_pos = [WIDTH // 2, 50]
    start_pos = [WIDTH // 2, HEIGHT - 50]
    
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
    
    pop_size = 800
    pop = Population(size=pop_size, mutation_rate=.02, start_pos=start_pos, target_pos=target_pos, dna_length=GEN_TTL)
    
    frame_count = 0
    gen = 1
    
    best_fitness_ever = 0.0
    best_path_ever = []
    stagnation = 0
    telemetry_history = []

    with open("swarm_telemetry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max_Fitness", "Avg_Fitness", "Successes", "Crashes", "Left_Faction", "Right_Faction", "Mutation_Rate"])
    
    max_gens = 50  # Default limit for headless run
    if len(sys.argv) > 1:
        try:
            max_gens = int(sys.argv[1])
        except ValueError:
            pass

    sim_speed = 5
    start_time = time.time()

    print(f"Starting headless simulation for {max_gens} generations...")

    while gen <= max_gens:
        for _ in range(sim_speed):
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
                telemetry_history.append((gen, max_fit, avg_fit, successes, crashes, left_count, right_count, pop.mutation_rate))
                if len(telemetry_history) > TELEMETRY_UI_ROWS:
                    telemetry_history.pop(0)

                with open("swarm_telemetry.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, round(max_fit, 2), round(avg_fit, 2), successes, crashes, left_count, right_count, pop.mutation_rate])

                if max_fit > best_fitness_ever:
                    best_fitness_ever = max_fit
                    best_bug = max(pop.creatures, key=lambda c: c.fitness)
                    best_path_ever = best_bug.path_history.copy()
                    stagnation = 0
                    pop.mutation_rate = 0.01
                else:
                    stagnation += 1

                if stagnation == 5:
                    print("*** Stagnation detected. Spiking mutation! ***")
                    pop.mutation_rate = 0.03
                elif stagnation > 7:
                    print("*** Cooling down mutation to stabilize swarm. ***")
                    pop.mutation_rate = 0.01
                    stagnation = 0

                print(f"--- Generation {gen} ---")
                print(f"Success: {successes}/{pop.size} | Crashed: {crashes}/{pop.size}")
                print(f"Max Fit: {max_fit:.5f} | Avg Fit: {avg_fit:.5f}\n")
                pop.natural_selection()
                frame_count = 0
                gen += 1
                print(f"Generation {gen} has begun...")
                if gen > max_gens:
                    break

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.4f} seconds.")
    if best_path_ever:
        print(f"Best path points recorded: {len(best_path_ever)}")

if __name__ == "__main__":
    main()

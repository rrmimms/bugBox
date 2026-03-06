import sys
import numpy as np
import time
# Try to import Population, path might need adjustment or ensure we run from root
from src.population import Population

WIDTH, HEIGHT = 800, 600
GEN_TTL = 800

class SimpleRect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def collidepoint(self, x, y):
        return (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h)

def main():
    target_pos = np.array([WIDTH // 2, 50])
    start_pos = np.array([WIDTH // 2, HEIGHT - 50])
    
    # Zig-zag obstacles using SimpleRect
    obstacles = [
        SimpleRect(0, 400, 600, 20),      # Left wall (gap on right)
        SimpleRect(200, 200, 600, 20)     # Right wall (gap on left)
    ]
    
    pop_size = 1000
    pop = Population(size=pop_size, mutation_rate=.02, start_pos=start_pos, target_pos=target_pos, dna_length=GEN_TTL)
    
    frame_count = 0
    gen = 1
    
    best_fitness_ever = 0.0
    stagnation = 0
    
    max_gens = 50  # Default limit for headless run
    if len(sys.argv) > 1:
        try:
            max_gens = int(sys.argv[1])
        except ValueError:
            pass

    start_time = time.time()

    print(f"Starting headless simulation for {max_gens} generations...")

    while gen <= max_gens:
        pop.update(frame_count, WIDTH, HEIGHT, obstacles)
        frame_count += 1
        
        if all(c.crashed or c.reached_goal for c in pop.creatures):
            frame_count = GEN_TTL
            
        if frame_count >= GEN_TTL:
            pop.evaluate_fitness()
            successes = sum(1 for c in pop.creatures if c.reached_goal)
            crashes = sum(1 for c in pop.creatures if c.crashed)
            max_fit = np.max(np.array([c.fitness for c in pop.creatures]))
            avg_fit = np.mean(np.array([c.fitness for c in pop.creatures]))
            
            # --- TRUE ADAPTIVE MUTATION ---
            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                stagnation = 0
                pop.mutation_rate = 0.01 
            else:
                stagnation += 1
                
            if stagnation == 5:
                print(f"*** Stagnation detected. Spiking mutation! ***")
                pop.mutation_rate = 0.05
            elif stagnation > 7:
                print(f"*** Cooling down mutation to stabilize swarm. ***")
                pop.mutation_rate = 0.01
                stagnation = 0  # Reset the clock to give them time to optimize
            # ------------------------------
                
            print(f"--- Generation {gen} ---")
            print(f"Success: {successes}/{pop.size} | Crashed: {crashes}/{pop.size}")
            print(f"Max Fit: {max_fit:.5f} | Avg Fit: {avg_fit:.5f}")
            
            pop.natural_selection()
            frame_count = 0
            gen += 1

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()

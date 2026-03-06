import pygame
import sys
import csv
import numpy as np
import math
from src.population import Population

BACKGROUND_COLOR = (22, 24, 32)
OBSTACLE_COLOR = (93, 106, 158)
TARGET_COLOR = (144, 182, 232)
HUD_TEXT_COLOR = (176, 214, 190)
BEST_PATH_COLOR = (233, 193, 112)

WIDTH, HEIGHT = 800, 600

FPS = 60

GEN_TTL = 500

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("bugBox")
    clock = pygame.time.Clock()
    
    target_pos = np.array([WIDTH // 2, 50])
    start_pos = np.array([WIDTH // 2, HEIGHT - 50])
    
    # Zig-zag obstacles
    # The Full Labyrinth
    # The Spacious Labyrinth
    # The "City Blocks" Layout
    static_obstacles = [
        # Bottom Row: Two blocks leaving a wide 200px alley in the dead center
        pygame.Rect(100, 400, 200, 80),
        pygame.Rect(500, 400, 200, 80),
        
        # Middle Row: A massive central block forcing the swarm to split left or right
        pygame.Rect(250, 250, 300, 80),
        
        # Top Row: Two guard blocks forcing the outer bugs back into the center
        pygame.Rect(50, 100, 200, 50),
        pygame.Rect(550, 100, 200, 50)
    ]
    
    pop = Population(size = 1000, mutation_rate=.02, start_pos=start_pos, target_pos=target_pos, dna_length=GEN_TTL)
    
    frame_count = 0
    gen = 1
    
    best_fitness_ever = 0.0
    best_path_ever = []
    stagnation = 0
    current_obstacles = static_obstacles

    with open("swarm_telemetry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max_Fitness", "Avg_Fitness", "Successes", "Crashes", "Left_Faction", "Right_Faction", "Mutation_Rate"])
    
    running = True
    
    SIM_SPEED = 5

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for _ in range(SIM_SPEED):
            door_y = 100 + math.sin(frame_count * 0.05) * 40
            moving_door_one = pygame.Rect(300, int(door_y), 20, 100)
            moving_door_two = pygame.Rect(475, int(door_y), 20, 100)
            current_obstacles = static_obstacles + [moving_door_one] + [moving_door_two]
            
            pop.update(frame_count, WIDTH, HEIGHT, current_obstacles)
            frame_count += 1
            
            if all(c.crashed or c.reached_goal for c in pop.creatures):
                frame_count = GEN_TTL
                
            if frame_count >= GEN_TTL:
                pop.evaluate_fitness()
                successes = sum(1 for c in pop.creatures if c.reached_goal)
                crashes = sum(1 for c in pop.creatures if c.crashed)
                max_fit = np.max(np.array([c.fitness for c in pop.creatures]))
                avg_fit = np.mean(np.array([c.fitness for c in pop.creatures]))
                left_count = sum(1 for c in pop.creatures if getattr(c, 'avg_x', 400) < 400)
                right_count = pop.size - left_count

                with open("swarm_telemetry.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, round(max_fit, 2), round(avg_fit, 2), successes, crashes, left_count, right_count, pop.mutation_rate])
                
               # --- TRUE ADAPTIVE MUTATION ---
                if max_fit > best_fitness_ever:
                    best_fitness_ever = max_fit
                    best_bug = max(pop.creatures, key=lambda c: c.fitness)
                    best_path_ever = best_bug.path_history.copy()
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
                print(f"Max Fit: {max_fit:.5f} | Avg Fit: {avg_fit:.5f}\n")
                pop.natural_selection()
                frame_count = 0
                gen += 1
                print (f"Generation {gen} has begun...")
        
        screen.fill(BACKGROUND_COLOR)
        if len(best_path_ever) > 1:
            pygame.draw.lines(screen, BEST_PATH_COLOR, False, best_path_ever, 3)
        pygame.draw.circle(screen, TARGET_COLOR, (int(target_pos[0]), int(target_pos[1])), 10)
        for obs in current_obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs)
        pop.draw(screen)
            
        font = pygame.font.SysFont("monospace", 15)
        stats = font.render(f"Gen: {gen} | Frame: {frame_count}", True, HUD_TEXT_COLOR)
        screen.blit(stats, (10, 10))
            
        pygame.display.flip()
        clock.tick(0)
            
    pygame.quit()
    sys.exit()
        
if __name__ == "__main__":
    main()
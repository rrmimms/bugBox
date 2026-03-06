import pygame
import sys
import csv
import math
from src.population import Population

BACKGROUND_COLOR = (22, 24, 32)
OBSTACLE_COLOR = (93, 106, 158)
TARGET_COLOR = (144, 182, 232)
HUD_TEXT_COLOR = (176, 214, 190)
BEST_PATH_COLOR = (233, 193, 112)
PANEL_BG_COLOR = (14, 16, 22)
PANEL_TABLE_BG_COLOR = (8, 10, 14)
PANEL_HEADER_COLOR = (245, 226, 162)
PANEL_TEXT_COLOR = (238, 244, 252)

SIM_WIDTH, HEIGHT = 800, 600
PANEL_WIDTH = 360
WIDTH = SIM_WIDTH + PANEL_WIDTH

FPS = 60

GEN_TTL = 800
TELEMETRY_UI_ROWS = 10

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("bugBox")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 15)
    small_font_12 = pygame.font.SysFont("monospace", 12)
    small_font_10 = pygame.font.SysFont("monospace", 10)
    
    target_pos = [SIM_WIDTH // 2, 50]
    start_pos = [SIM_WIDTH // 2, HEIGHT - 50]
    
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
    moving_door_one = pygame.Rect(300, 100, 20, 100)
    moving_door_two = pygame.Rect(475, 100, 20, 100)
    current_obstacles = static_obstacles + [moving_door_one, moving_door_two]
    
    pop = Population(size = 1000, mutation_rate=.02, start_pos=start_pos, target_pos=target_pos, dna_length=GEN_TTL)
    
    frame_count = 0
    gen = 1
    
    best_fitness_ever = 0.0
    best_path_ever = []
    stagnation = 0
    telemetry_history = []

    with open("swarm_telemetry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Max_Fitness", "Avg_Fitness", "Successes", "Crashes", "Left_Faction", "Right_Faction", "Mutation_Rate"])
    
    running = True
    
    SIM_SPEED = 5
    panel_x, panel_y = SIM_WIDTH + 10, 10
    panel_width = PANEL_WIDTH - 20
    headers = ["Gen", "MaxF", "AvgF", "Succ", "Crash", "Left", "Right", "Mut"]
    col_padding = 8
    left_padding = 8
    right_padding = 8
    telemetry_panel_surface = None
    telemetry_panel_height = 0
    telemetry_dirty = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for _ in range(SIM_SPEED):
            door_y = 100 + math.sin(frame_count * 0.05) * 40
            moving_door_one.y = int(door_y)
            moving_door_two.y = int(door_y)
            
            pop.update(frame_count, SIM_WIDTH, HEIGHT, current_obstacles)
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
                telemetry_dirty = True

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
        
        stats = font.render(f"Gen: {gen} | Frame: {frame_count}", True, HUD_TEXT_COLOR)
        screen.blit(stats, (10, 10))

        if telemetry_dirty:
            table_rows = [
                [
                    str(row[0]),
                    f"{row[1]:.2f}",
                    f"{row[2]:.2f}",
                    str(row[3]),
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                    f"{row[7]:.2f}",
                ]
                for row in reversed(telemetry_history)
            ]

            small_font = small_font_12
            col_widths = []
            for column_index, header in enumerate(headers):
                content_width = small_font.size(header)[0]
                for row in table_rows:
                    cell_width = small_font.size(row[column_index])[0]
                    if cell_width > content_width:
                        content_width = cell_width
                col_widths.append(content_width + col_padding)

            available_table_width = panel_width - left_padding - right_padding
            if sum(col_widths) > available_table_width:
                small_font = small_font_10
                col_widths = []
                for column_index, header in enumerate(headers):
                    content_width = small_font.size(header)[0]
                    for row in table_rows:
                        cell_width = small_font.size(row[column_index])[0]
                        if cell_width > content_width:
                            content_width = cell_width
                    col_widths.append(content_width + col_padding)

            row_height = small_font.get_height() + 2
            telemetry_panel_height = 14 + row_height + (len(table_rows) * row_height) + 10
            telemetry_panel_surface = pygame.Surface((panel_width, telemetry_panel_height), pygame.SRCALPHA)
            telemetry_panel_surface.fill((*PANEL_TABLE_BG_COLOR, 235))
            pygame.draw.rect(telemetry_panel_surface, PANEL_HEADER_COLOR, (0, 0, panel_width, telemetry_panel_height), 1)

            current_x = left_padding
            header_y = 8
            for column_index, header in enumerate(headers):
                header_surface = small_font.render(header, True, PANEL_HEADER_COLOR)
                telemetry_panel_surface.blit(header_surface, (current_x, header_y))
                current_x += col_widths[column_index]

            for row_index, row in enumerate(table_rows):
                row_y = 8 + row_height + (row_index * row_height)
                current_x = left_padding
                for column_index, value in enumerate(row):
                    value_surface = small_font.render(value, True, PANEL_TEXT_COLOR)
                    value_x = current_x + col_widths[column_index] - col_padding - value_surface.get_width()
                    telemetry_panel_surface.blit(value_surface, (value_x, row_y))
                    current_x += col_widths[column_index]

            telemetry_dirty = False

        pygame.draw.rect(screen, PANEL_BG_COLOR, (SIM_WIDTH, 0, PANEL_WIDTH, HEIGHT))
        pygame.draw.line(screen, PANEL_HEADER_COLOR, (SIM_WIDTH, 0), (SIM_WIDTH, HEIGHT), 2)
        if telemetry_panel_surface is not None:
            screen.blit(telemetry_panel_surface, (panel_x, panel_y))
            
        pygame.display.flip()
        clock.tick(FPS)
            
    pygame.quit()
    sys.exit()
        
if __name__ == "__main__":
    main()
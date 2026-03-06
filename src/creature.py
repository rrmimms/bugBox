try:
    import pygame
except ImportError:
    pygame = None
import math
import numpy as np
from src.dna import DNA
from src.nn import NeuralNet

CREATURE_DEFAULT_COLOR = (126, 166, 196)
CREATURE_GOAL_COLOR = (140, 201, 151)
CREATURE_CRASH_COLOR = (184, 112, 126)

class Creature:
    def __init__(self, start_pos, dna_source, is_elite=False):
        self.pos = np.array(start_pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.acc = np.zeros(2, dtype=np.float64)
        self.finish_time = 0

        self.brain = NeuralNet(input_size=10, hidden_size=8, output_size=2)
        
        if isinstance(dna_source, int):
            self.dna = DNA(self.brain.num_genes)
            self.dna.length = dna_source
        else:
            self.dna = dna_source

        if not hasattr(self.dna, "length"):
            self.dna.length = self.dna.num_genes

        genes = np.asarray(self.dna.genes, dtype=np.float64)
        if genes.size < self.brain.num_genes:
            genes = np.pad(genes, (0, self.brain.num_genes - genes.size), mode="constant")
        elif genes.size > self.brain.num_genes:
            genes = genes[:self.brain.num_genes]
        self.dna.genes = genes
        self.dna.num_genes = genes.size
        self.brain.set_dna(self.dna.genes)
            
        self.fitness = 0.0
        self.crashed = False
        self.reached_goal = False
        self.is_elite = is_elite
        self.path_history = []
        self.lifetime = 0
        self.closest_dist = float('inf')
        self.color: tuple[int, int, int] = CREATURE_DEFAULT_COLOR
        
    def apply_force(self, force):
        self.acc += force

    def get_sensor_data(self, obstacles, target_pos):
        ray_length = 100.0
        pos_x = float(self.pos[0])
        pos_y = float(self.pos[1])

        to_target = target_pos - self.pos
        target_dist = math.hypot(float(to_target[0]), float(to_target[1]))
        if target_dist > 0:
            direction = to_target / target_dist
            dir_x = float(direction[0])
            dir_y = float(direction[1])
        else:
            dir_x = 0.0
            dir_y = 0.0

        def normalized_ray_distance(end_offset):
            start = (int(pos_x), int(pos_y))
            end = (int(pos_x + end_offset[0]), int(pos_y + end_offset[1]))
            closest = ray_length

            if obstacles is None:
                return 1.0

            for obs in obstacles:
                if not hasattr(obs, "clipline"):
                    continue
                hit = obs.clipline(start, end)
                if hit:
                    hit_points = hit if isinstance(hit, tuple) else tuple(hit)
                    if len(hit_points) >= 2:
                        hit_x, hit_y = hit_points[0]
                        dist = math.hypot(float(hit_x) - pos_x, float(hit_y) - pos_y)
                        if dist < closest:
                            closest = dist

            return float(np.clip(closest / ray_length, 0.0, 1.0))

       # The 4 cardinal directions (Up, Down, Left, Right)
        dist_up = normalized_ray_distance((0, -ray_length))
        dist_down = normalized_ray_distance((0, ray_length))
        dist_left = normalized_ray_distance((-ray_length, 0))
        dist_right = normalized_ray_distance((ray_length, 0))
        
        # NEW: The 4 diagonal directions (45 degrees)
        diag = ray_length * 0.7071
        dist_ul = normalized_ray_distance((-diag, -diag))
        dist_ur = normalized_ray_distance((diag, -diag))
        dist_dl = normalized_ray_distance((-diag, diag))
        dist_dr = normalized_ray_distance((diag, diag))

        # Return all 10 inputs!
        return [dir_x, dir_y, dist_up, dist_down, dist_left, dist_right, dist_ul, dist_ur, dist_dl, dist_dr]
            
    def update(self, tick, target_pos, width, height, obstacles=None):
        if self.crashed or self.reached_goal:
            return

        delta = target_pos - self.pos
        dist_to_target = math.hypot(float(delta[0]), float(delta[1]))
        self.closest_dist = min(self.closest_dist, dist_to_target)

        if dist_to_target < 15:
            self.reached_goal = True
            self.pos = target_pos.copy()
            self.finish_time = tick
            return

        if (self.pos[0] < 0 or self.pos[0] > width
                or self.pos[1] < 0 or self.pos[1] > height):
            self.crashed = True
            return

        if obstacles is not None:
            for obs in obstacles:
                if obs.collidepoint(self.pos[0], self.pos[1]):
                    self.crashed = True
                    return

        self.path_history.append((int(self.pos[0]), int(self.pos[1])))
        inputs = self.get_sensor_data(obstacles, target_pos)
        
        # --- FIX 3A: Track reckless driving ---
        # Indices 2 through 5 are the raycast distances
        min_sensor = min(inputs[2:]) 
        if not hasattr(self, 'min_wall_dist'):
            self.min_wall_dist = 1.0
        self.min_wall_dist = min(self.min_wall_dist, min_sensor)
        # --------------------------------------
        
        force = self.brain.forward(inputs)
        self.apply_force(force)
        self.vel += self.acc

        speed = math.hypot(float(self.vel[0]), float(self.vel[1]))
        if speed > 6.0:
            self.vel = (self.vel / speed) * 6.0

        self.pos += self.vel
        self.acc[:] = 0
        self.lifetime = tick
                
    def calc_fitness(self, target_pos):
        if len(self.path_history) > 0:
            self.avg_x = sum(p[0] for p in self.path_history) / len(self.path_history)
        else:
            self.avg_x = self.pos[0]
            
        dist = max(self.closest_dist, 1.0)
            
        proximity_score = 10000.0 / dist
        
        # FIX 1: Use 2000.0 instead of self.dna.length
        survival_score = self.lifetime / 2000.0
        self.fitness = proximity_score + survival_score
            
        if self.reached_goal:
            # FIX 2: Use 2000.0 instead of self.dna.length
            speed_mult = (2000.0 - self.finish_time) / 2000.0
            self.fitness = 10000.0 + (speed_mult * 10000.0)
                
        elif self.crashed:
            self.fitness *= 0.9
            
        safety_multiplier = 0.8 + (0.2 * getattr(self, 'min_wall_dist', 1.0))
        self.fitness *= safety_multiplier
                
    def draw(self, screen):
        if pygame is None:
            return

        if self.is_elite and len(self.path_history) > 1:
            pygame.draw.lines(screen, self.color, False, self.path_history, 2)
            
        if self.reached_goal:
            color = CREATURE_GOAL_COLOR
        elif self.crashed:
            color = CREATURE_CRASH_COLOR
        else:
            color = self.color
                
        pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), 5)
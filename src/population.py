import numpy as np
from src.creature import Creature

class Population:
    def __init__(self, size, mutation_rate, start_pos, target_pos, dna_length):
        self.size = size
        self.mutation_rate = mutation_rate
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.dna_length = dna_length

        self.colors = {
            "left": (78, 156, 138),
            "left_elite": (102, 181, 162),
            "right": (214, 120, 98),
            "right_elite": (230, 142, 122),
            "champion": (237, 196, 106),
        }
        
        self.creatures = [Creature(self.start_pos, self.dna_length) for _ in range(size)]
        
    def update(self, tick, width, height, obstacles=None):
        for creature in self.creatures:
            if creature.crashed or creature.reached_goal:
                continue
            creature.update(tick, self.target_pos, width, height, obstacles)
            
    def draw(self, screen):
        for creature in self.creatures:
            creature.draw(screen)
            
    def evaluate_fitness(self):
        for creature in self.creatures:
            creature.calc_fitness(self.target_pos)
            
    # ...existing code...
    def natural_selection(self):
        # 1. Speciation: Sort bugs into Factions
        left_faction = [c for c in self.creatures if getattr(c, 'avg_x', 400) < 400]
        right_faction = [c for c in self.creatures if getattr(c, 'avg_x', 400) >= 400]

        if not left_faction:
            left_faction = right_faction
        if not right_faction:
            right_faction = left_faction

        # All-time Champion
        best_overall = max(self.creatures, key=lambda c: c.fitness)

        left_score = sum(c.fitness for c in left_faction)
        right_score = sum(c.fitness for c in right_faction)
        total_score = left_score + right_score

        left_alloc = int(self.size * (left_score / total_score)) if total_score > 0 else self.size // 2
        right_alloc = self.size - left_alloc

        new_creatures = []

        def select_parent(faction):
            if not faction:
                raise ValueError("Cannot select parent from empty faction")
            k = min(8, len(faction))  # prevents np.random.choice crash on small factions
            tournament = np.random.choice(faction, size=k, replace=False)
            return max(tournament, key=lambda c: c.fitness)

        # 2. Breed the Left Faction (GREEN)
        left_faction.sort(key=lambda x: x.fitness, reverse=True)
        elite_left = 0 if left_alloc == 0 else min(
            len(left_faction),
            left_alloc,
            max(1, int(left_alloc * 0.10))
        )

        for i in range(elite_left):
            elite_dna = type(left_faction[0].dna)(
                left_faction[i].brain.num_genes,  # <-- Corrected: Ask the brain for its exact size!
                left_faction[i].dna.max_force,
                existing_genes=left_faction[i].dna.genes.copy()
            )
            new_bug = type(left_faction[0])(self.start_pos.copy(), elite_dna, is_elite=True)
            new_bug.color = self.colors["champion"] if left_faction[i] == best_overall else self.colors["left_elite"]
            new_creatures.append(new_bug)

        while len(new_creatures) < left_alloc:
            pA = select_parent(left_faction)
            pB = select_parent(left_faction)
            child_dna = pA.dna.crossover(pB.dna, pA.fitness, pB.fitness)
            child_dna.mutate(self.mutation_rate)
            new_child = type(pA)(self.start_pos.copy(), child_dna)
            new_child.color = self.colors["left"]
            new_creatures.append(new_child)

        # 3. Breed the Right Faction (RED)
        right_faction.sort(key=lambda x: x.fitness, reverse=True)
        elite_right = 0 if right_alloc == 0 else min(
            len(right_faction),
            right_alloc,
            max(1, int(right_alloc * 0.10))
        )

        for i in range(elite_right):
            elite_dna = type(right_faction[0].dna)(
                right_faction[i].brain.num_genes,
                right_faction[i].dna.max_force,
                existing_genes=right_faction[i].dna.genes.copy()
            )
            new_bug = type(right_faction[0])(self.start_pos.copy(), elite_dna, is_elite=True)
            new_bug.color = self.colors["champion"] if right_faction[i] == best_overall else self.colors["right_elite"]
            new_creatures.append(new_bug)

        while len(new_creatures) < self.size:
            pA = select_parent(right_faction)
            pB = select_parent(right_faction)
            child_dna = pA.dna.crossover(pB.dna, pA.fitness, pB.fitness)
            child_dna.mutate(self.mutation_rate)
            new_child = type(pA)(self.start_pos.copy(), child_dna)
            new_child.color = self.colors["right"]
            new_creatures.append(new_child)

        self.creatures = new_creatures

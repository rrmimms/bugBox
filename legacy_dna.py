import numpy as np

class DNA:
    def __init__(self, length, max_force=.5, existing_genes=None):
        self.length = length
        self.max_force = max_force
        
        if existing_genes is not None:
            self.genes = existing_genes
            
        else:
            angles = np.random.uniform(0, 2*np.pi, length)
            self.genes = np.column_stack((np.cos(angles), np.sin(angles))) * max_force
            
    def crossover(self, partner, my_fitness, partner_fitness):
        total_fit = my_fitness + partner_fitness
        if total_fit == 0:
            prob = 0.5
        else:
            prob = my_fitness / total_fit
            
        mask = np.random.rand(self.length) < prob
        child_genes = np.where(mask[:, np.newaxis], self.genes, partner.genes)
        
        return DNA(self.length, self.max_force, existing_genes=child_genes)
    
    def mutate(self, mutation_rate=.01):
        mutations = np.random.rand(self.length) < mutation_rate
        num_mutations = int(np.sum(mutations))
        
        if num_mutations > 0:
            noise = np.random.uniform(-0.05, 0.05, (num_mutations, 2))
            self.genes[mutations] += noise
            
            mags = np.linalg.norm(self.genes[mutations], axis=1, keepdims=True)
            mags = np.maximum(mags, 1e-8)
            self.genes[mutations] = (self.genes[mutations] / mags) * self.max_force
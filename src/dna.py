import numpy as np
from numpy.typing import NDArray

class DNA:
    def __init__(self, num_genes, max_force=0.5, existing_genes=None):
        self.num_genes = num_genes
        self.length = num_genes
        self.max_force = max_force
        self.genes: NDArray[np.float64]
        
        if existing_genes is not None:
            # Force Pylance to recognize this as a NumPy Array
            self.genes = np.array(existing_genes, dtype=np.float64)
        else:
            # Using the explicit 'size=' keyword tells Pylance it returns an Array
            self.genes = np.array(
                np.random.uniform(-1.0, 1.0, size=self.num_genes),
                dtype=np.float64,
            )
    def crossover(self, partner, fitA, fitB):
        child_genes = np.copy(self.genes)
        # Randomly inherit each brain connection from either parent
        mask = np.random.rand(self.num_genes) > 0.5
        child_genes[mask] = partner.genes[mask]
        return type(self)(self.num_genes, self.max_force, existing_genes=child_genes)

    def mutate(self, mutation_rate):
        # Vectorized mutation: No more 'for' loops! This is lightning fast.
        # 1. Create a true/false mask for genes that should mutate
        mask = np.random.rand(self.num_genes) < mutation_rate
        
        # 2. Add random noise only to the genes where the mask is True
        num_mutations = np.sum(mask)
        if num_mutations > 0:
            self.genes[mask] += np.random.uniform(-0.2, 0.2, size=num_mutations)
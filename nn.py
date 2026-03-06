import numpy as np

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases (This is what we will mutate!)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.randn(self.output_size)
        
        # Calculate total number of "genes" needed to build this brain
        self.num_genes = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

    def forward(self, inputs):
        # Hidden layer with Tanh activation
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = np.tanh(z1)
        
        # Output layer with Tanh activation (Outputs -1.0 to +1.0 for X and Y steering)
        z2 = np.dot(a1, self.W2) + self.b2
        output = np.tanh(z2)
        
        return output

    def get_dna(self):
        # Flatten all weights and biases into a single 1D array to pass to the genetic algorithm
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])

    def set_dna(self, dna):
        # Unpack a 1D DNA array back into the 2D weight matrices
        start = 0
        
        end = self.input_size * self.hidden_size
        self.W1 = dna[start:end].reshape(self.input_size, self.hidden_size)
        start = end
        
        end = start + self.hidden_size
        self.b1 = dna[start:end]
        start = end
        
        end = start + (self.hidden_size * self.output_size)
        self.W2 = dna[start:end].reshape(self.hidden_size, self.output_size)
        start = end
        
        self.b2 = dna[start:]
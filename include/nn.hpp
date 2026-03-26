#pragma once
#include <vector>

class NeuralNet {
public:
    int input_size;
    int hidden_size;
    int output_size;
    int num_genes;

    // Flattened weights and biases
    std::vector<double> W1;
    std::vector<double> b1;
    std::vector<double> W2;
    std::vector<double> b2;

    NeuralNet(int input_size, int hidden_size, int output_size);
    
    std::vector<double> forward(const std::vector<double>& inputs) const;
    std::vector<double> get_dna() const;
    void set_dna(const std::vector<double>& dna);
};

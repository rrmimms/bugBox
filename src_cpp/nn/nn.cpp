#include "nn.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace {
    std::mt19937& get_rng() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }
}

NeuralNet::NeuralNet(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    
    num_genes = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size;
    
    std::normal_distribution<double> dis(0.0, 1.0);
    
    W1.resize(input_size * hidden_size);
    for(auto& w : W1) w = dis(get_rng());
    
    b1.resize(hidden_size);
    for(auto& w : b1) w = dis(get_rng());
    
    W2.resize(hidden_size * output_size);
    for(auto& w : W2) w = dis(get_rng());
    
    b2.resize(output_size);
    for(auto& w : b2) w = dis(get_rng());
}

std::vector<double> NeuralNet::forward(const std::vector<double>& inputs) const {
    std::vector<double> a1(hidden_size, 0.0);
    for (int j = 0; j < hidden_size; ++j) {
        double z1 = b1[j];
        for (int i = 0; i < input_size; ++i) {
            z1 += inputs[i] * W1[i * hidden_size + j];
        }
        a1[j] = std::tanh(z1);
    }
    
    std::vector<double> output(output_size, 0.0);
    for (int j = 0; j < output_size; ++j) {
        double z2 = b2[j];
        for (int i = 0; i < hidden_size; ++i) {
            z2 += a1[i] * W2[i * output_size + j];
        }
        output[j] = std::tanh(z2);
    }
    
    return output;
}

std::vector<double> NeuralNet::get_dna() const {
    std::vector<double> dna;
    dna.reserve(num_genes);
    dna.insert(dna.end(), W1.begin(), W1.end());
    dna.insert(dna.end(), b1.begin(), b1.end());
    dna.insert(dna.end(), W2.begin(), W2.end());
    dna.insert(dna.end(), b2.begin(), b2.end());
    return dna;
}

void NeuralNet::set_dna(const std::vector<double>& dna) {
    int start = 0;
    int end = input_size * hidden_size;
    std::copy(dna.begin() + start, dna.begin() + end, W1.begin());
    
    start = end;
    end = start + hidden_size;
    std::copy(dna.begin() + start, dna.begin() + end, b1.begin());
    
    start = end;
    end = start + (hidden_size * output_size);
    std::copy(dna.begin() + start, dna.begin() + end, W2.begin());
    
    start = end;
    std::copy(dna.begin() + start, dna.end(), b2.begin());
}

#pragma once
#include <vector>
#include <random>

class DNA {
public:
    int num_genes;
    double max_force;
    std::vector<double> genes;

    DNA(int num_genes, double max_force = 0.5, const std::vector<double>* existing_genes = nullptr);
    DNA() : num_genes(0), max_force(0.5) {}

    DNA crossover(const DNA& partner, double fitA, double fitB) const;
    void mutate(double mutation_rate);

private:
    static std::mt19937& get_rng();
};

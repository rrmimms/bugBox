#include "dna.hpp"
#include <algorithm>

std::mt19937& DNA::get_rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

DNA::DNA(int num_genes, double max_force, const std::vector<double>* existing_genes)
    : num_genes(num_genes), max_force(max_force) {
    if (existing_genes) {
        genes = *existing_genes;
    } else {
        genes.resize(num_genes);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for(int i=0; i<num_genes; ++i) {
            genes[i] = dis(get_rng());
        }
    }
}

DNA DNA::crossover(const DNA& partner, double fitA, double fitB) const {
    std::vector<double> child_genes(num_genes);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(int i=0; i<num_genes; ++i) {
        if (dis(get_rng()) > 0.5) {
            child_genes[i] = partner.genes[i];
        } else {
            child_genes[i] = genes[i];
        }
    }
    return DNA(num_genes, max_force, &child_genes);
}

void DNA::mutate(double mutation_rate) {
    std::uniform_real_distribution<> rate_dis(0.0, 1.0);
    std::uniform_real_distribution<> mut_dis(-0.2, 0.2);
    for(int i=0; i<num_genes; ++i) {
        if (rate_dis(get_rng()) < mutation_rate) {
            genes[i] += mut_dis(get_rng());
        }
    }
}

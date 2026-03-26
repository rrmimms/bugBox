#include "population.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

Population::Population(int size, double mutation_rate, Point start_pos, Point target_pos) 
    : size(size), mutation_rate(mutation_rate), start_pos(start_pos), target_pos(target_pos) {
    
    colors["left"] = {78, 156, 138};
    colors["left_elite"] = {102, 181, 162};
    colors["right"] = {214, 120, 98};
    colors["right_elite"] = {230, 142, 122};
    colors["champion"] = {237, 196, 106};

    for (int i = 0; i < size; ++i) {
        creatures.push_back(std::make_shared<Creature>(start_pos, nullptr));
    }
}

void Population::update(int tick, double width, double height, const std::vector<Obstacle>& obstacles) {
    for (auto& creature : creatures) {
        if (!creature->crashed && !creature->reached_goal) {
            creature->update(tick, target_pos, width, height, obstacles);
        }
    }
}

void Population::evaluate_fitness() {
    for (auto& creature : creatures) {
        creature->calc_fitness(target_pos);
    }
}

namespace {
    std::mt19937& get_rng() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }
}

void Population::natural_selection() {
    std::vector<std::shared_ptr<Creature>> left_faction;
    std::vector<std::shared_ptr<Creature>> right_faction;

    for (const auto& c : creatures) {
        if (c->avg_x < 400.0) {
            left_faction.push_back(c);
        } else {
            right_faction.push_back(c);
        }
    }

    if (left_faction.empty()) left_faction = right_faction;
    if (right_faction.empty()) right_faction = left_faction;

    // Best overall
    auto best_overall = *std::max_element(creatures.begin(), creatures.end(), 
        [](const std::shared_ptr<Creature>& a, const std::shared_ptr<Creature>& b) {
            return a->fitness < b->fitness;
        });

    double left_score = 0.0;
    for (const auto& c : left_faction) left_score += c->fitness;
    
    double right_score = 0.0;
    for (const auto& c : right_faction) right_score += c->fitness;
    
    double total_score = left_score + right_score;

    int raw_left = total_score > 0.0 ? (int)(size * (left_score / total_score)) : size / 2;
    int min_pop = (int)(size * 0.10);

    int left_alloc = std::max(min_pop, std::min(size - min_pop, raw_left));
    int right_alloc = size - left_alloc;

    std::vector<std::shared_ptr<Creature>> new_creatures;

    auto select_parent = [](const std::vector<std::shared_ptr<Creature>>& faction) -> std::shared_ptr<Creature> {
        if (faction.empty()) throw std::runtime_error("Empty faction");
        int k = std::min(8, (int)faction.size());
        
        std::vector<std::shared_ptr<Creature>> tournament;
        std::sample(faction.begin(), faction.end(), std::back_inserter(tournament), k, get_rng());
        
        return *std::max_element(tournament.begin(), tournament.end(),
            [](const std::shared_ptr<Creature>& a, const std::shared_ptr<Creature>& b) {
                return a->fitness < b->fitness;
            });
    };

    // Sort descending by fitness
    auto comp = [](const std::shared_ptr<Creature>& a, const std::shared_ptr<Creature>& b) {
        return a->fitness > b->fitness;
    };

    std::sort(left_faction.begin(), left_faction.end(), comp);
    int elite_left = left_alloc == 0 ? 0 : std::min({(int)left_faction.size(), left_alloc, std::max(1, (int)(left_alloc * 0.10))});

    for (int i = 0; i < elite_left; ++i) {
        auto elite_dna = std::make_shared<DNA>(
            left_faction[i]->brain.num_genes,
            left_faction[i]->dna->max_force,
            &(left_faction[i]->dna->genes)
        );
        auto new_bug = std::make_shared<Creature>(start_pos, elite_dna, true);
        auto c = (left_faction[i] == best_overall) ? colors["champion"] : colors["left_elite"];
        new_bug->r = c.r; new_bug->g = c.g; new_bug->b = c.b;
        new_creatures.push_back(new_bug);
    }

    while (new_creatures.size() < (size_t)left_alloc) {
        auto pA = select_parent(left_faction);
        auto pB = select_parent(left_faction);
        auto child_dna_val = pA->dna->crossover(*(pB->dna), pA->fitness, pB->fitness);
        child_dna_val.mutate(mutation_rate);
        
        auto new_child = std::make_shared<Creature>(start_pos, std::make_shared<DNA>(child_dna_val));
        PopColor c = colors["left"];
        new_child->r = c.r; new_child->g = c.g; new_child->b = c.b;
        new_creatures.push_back(new_child);
    }

    std::sort(right_faction.begin(), right_faction.end(), comp);
    int elite_right = right_alloc == 0 ? 0 : std::min({(int)right_faction.size(), right_alloc, std::max(1, (int)(right_alloc * 0.10))});

    for (int i = 0; i < elite_right; ++i) {
        auto elite_dna = std::make_shared<DNA>(
            right_faction[i]->brain.num_genes,
            right_faction[i]->dna->max_force,
            &(right_faction[i]->dna->genes)
        );
        auto new_bug = std::make_shared<Creature>(start_pos, elite_dna, true);
        auto c = (right_faction[i] == best_overall) ? colors["champion"] : colors["right_elite"];
        new_bug->r = c.r; new_bug->g = c.g; new_bug->b = c.b;
        new_creatures.push_back(new_bug);
    }

    while (new_creatures.size() < (size_t)size) {
        auto pA = select_parent(right_faction);
        auto pB = select_parent(right_faction);
        auto child_dna_val = pA->dna->crossover(*(pB->dna), pA->fitness, pB->fitness);
        child_dna_val.mutate(mutation_rate);
        
        auto new_child = std::make_shared<Creature>(start_pos, std::make_shared<DNA>(child_dna_val));
        PopColor c = colors["right"];
        new_child->r = c.r; new_child->g = c.g; new_child->b = c.b;
        new_creatures.push_back(new_child);
    }

    creatures = new_creatures;
}

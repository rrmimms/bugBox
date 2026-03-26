#pragma once

#include "creature.hpp"
#include <vector>
#include <memory>
#include <map>
#include <string>

struct PopColor {
    int r, g, b;
};

class Population {
public:
    int size;
    double mutation_rate;
    Point start_pos;
    Point target_pos;

    std::vector<std::shared_ptr<Creature>> creatures;
    std::map<std::string, PopColor> colors;

    Population(int size, double mutation_rate, Point start_pos, Point target_pos);

    void update(int tick, double width, double height, const std::vector<Obstacle>& obstacles);
    void evaluate_fitness();
    void natural_selection();
};

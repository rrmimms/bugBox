#include "creature.hpp"
#include <algorithm>

Creature::Creature(Point start_pos, std::shared_ptr<DNA> dna_source, bool is_elite)
    : brain(10, 8, 2), dna(dna_source), is_elite(is_elite) {
    
    pos[0] = start_pos.x;
    pos[1] = start_pos.y;
    vel[0] = 0.0; vel[1] = 0.0;
    acc[0] = 0.0; acc[1] = 0.0;
    finish_time = 0;
    
    if (!dna) {
        dna = std::make_shared<DNA>(brain.num_genes);
    }
    
    if (dna->genes.size() < (size_t)brain.num_genes) {
        dna->genes.resize(brain.num_genes, 0.0);
    } else if (dna->genes.size() > (size_t)brain.num_genes) {
        dna->genes.resize(brain.num_genes);
    }
    dna->num_genes = brain.num_genes;
    
    brain.set_dna(dna->genes);
    
    fitness = 0.0;
    crashed = false;
    reached_goal = false;
    lifetime = 0;
    closest_dist = std::numeric_limits<double>::infinity();
    min_wall_dist = 1.0;
    avg_x = 0.0;
    
    r = 126; g = 166; b = 196; // CREATURE_DEFAULT_COLOR
}

void Creature::apply_force(const std::vector<double>& force) {
    if (force.size() >= 2) {
        acc[0] += force[0];
        acc[1] += force[1];
    }
}

std::vector<double> Creature::get_sensor_data(const std::vector<Obstacle>& obstacles, Point target_pos) {
    double ray_length = 100.0;
    
    double to_target_x = target_pos.x - pos[0];
    double to_target_y = target_pos.y - pos[1];
    double target_dist = std::hypot(to_target_x, to_target_y);
    
    double dir_x = 0.0, dir_y = 0.0;
    if (target_dist > 0.0) {
        dir_x = to_target_x / target_dist;
        dir_y = to_target_y / target_dist;
    }
    
    auto normalized_ray_distance = [&](double off_x, double off_y) {
        double closest_t = 1.0;
        double ex = pos[0] + off_x;
        double ey = pos[1] + off_y;
        
        for (const auto& obs : obstacles) {
            double t = obs.ray_cast(pos[0], pos[1], ex, ey);
            if (t < closest_t) {
                closest_t = t;
            }
        }
        return closest_t;
    };
    
    double dist_up = normalized_ray_distance(0, -ray_length);
    double dist_down = normalized_ray_distance(0, ray_length);
    double dist_left = normalized_ray_distance(-ray_length, 0);
    double dist_right = normalized_ray_distance(ray_length, 0);
    
    double diag = ray_length * 0.7071;
    double dist_ul = normalized_ray_distance(-diag, -diag);
    double dist_ur = normalized_ray_distance(diag, -diag);
    double dist_dl = normalized_ray_distance(-diag, diag);
    double dist_dr = normalized_ray_distance(diag, diag);
    
    return {dir_x, dir_y, dist_up, dist_down, dist_left, dist_right, dist_ul, dist_ur, dist_dl, dist_dr};
}

void Creature::update(int tick, Point target_pos, double width, double height, const std::vector<Obstacle>& obstacles) {
    if (crashed || reached_goal) return;
    
    double dist_to_target = std::hypot(target_pos.x - pos[0], target_pos.y - pos[1]);
    closest_dist = std::min(closest_dist, dist_to_target);
    
    if (dist_to_target < 15.0) {
        reached_goal = true;
        pos[0] = target_pos.x;
        pos[1] = target_pos.y;
        finish_time = tick;
        return;
    }
    
    if (pos[0] < 0 || pos[0] > width || pos[1] < 0 || pos[1] > height) {
        crashed = true;
        return;
    }
    
    for (const auto& obs : obstacles) {
        if (obs.collidepoint(pos[0], pos[1])) {
            crashed = true;
            return;
        }
    }
    
    path_history.push_back({pos[0], pos[1]});
    
    std::vector<double> inputs = get_sensor_data(obstacles, target_pos);
    
    double min_sensor = 1.0;
    for (size_t i = 2; i < inputs.size(); ++i) { // indices 2 through 9 are rays
        if (inputs[i] < min_sensor) min_sensor = inputs[i];
    }
    min_wall_dist = std::min(min_wall_dist, min_sensor);
    
    std::vector<double> force = brain.forward(inputs);
    apply_force(force);
    
    vel[0] += acc[0];
    vel[1] += acc[1];
    
    double speed = std::hypot(vel[0], vel[1]);
    if (speed > 6.0) {
        vel[0] = (vel[0] / speed) * 6.0;
        vel[1] = (vel[1] / speed) * 6.0;
    }
    
    pos[0] += vel[0];
    pos[1] += vel[1];
    
    acc[0] = 0.0;
    acc[1] = 0.0;
    lifetime = tick;
}

void Creature::calc_fitness(Point target_pos) {
    if (!path_history.empty()) {
        double sum_x = 0;
        for (const auto& p : path_history) sum_x += p.x;
        avg_x = sum_x / path_history.size();
    } else {
        avg_x = pos[0];
    }
    
    double dist = std::max(closest_dist, 1.0);
    double proximity_score = 10000.0 / dist;
    double survival_score = lifetime / 2000.0;
    
    fitness = proximity_score + survival_score;
    
    if (reached_goal) {
        double speed_mult = (2000.0 - finish_time) / 2000.0;
        fitness = 10000.0 + (speed_mult * 10000.0);
    } else if (crashed) {
        fitness *= 0.9;
    }
    
    double safety_multiplier = 0.8 + (0.2 * min_wall_dist);
    fitness *= safety_multiplier;
}

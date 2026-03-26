#pragma once

#include "dna.hpp"
#include "nn.hpp"
#include <vector>
#include <tuple>
#include <memory>
#include <cmath>

struct Point {
    double x;
    double y;
};

struct Obstacle {
    double x, y, width, height;
    
    bool collidepoint(double px, double py) const {
        return px >= x && px <= x + width && py >= y && py <= y + height;
    }

    // Very simple ray intersection for line segments (start to end)
    // Returns distance ratio (0.0 to 1.0) along the ray, or 1.0 if no hit.
    double ray_cast(double sx, double sy, double ex, double ey) const {
        // Line from (sx, sy) to (ex, ey)
        // Rect from (x, y) to (x+width, y+height)
        
        // This is a simplified AABB raycast. 
        // We evaluate against 4 segments of the rectangle. 
        std::vector<std::pair<Point, Point>> edges = {
            {{x, y}, {x + width, y}},
            {{x + width, y}, {x + width, y + height}},
            {{x + width, y + height}, {x, y + height}},
            {{x, y + height}, {x, y}}
        };

        double min_t = 1.0;
        for (const auto& edge : edges) {
            double x1 = sx, y1 = sy, x2 = ex, y2 = ey;
            double x3 = edge.first.x, y3 = edge.first.y, x4 = edge.second.x, y4 = edge.second.y;
            
            double den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if (std::abs(den) < 1e-6) continue;
            
            double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
            double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;
            
            if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
                if (t < min_t) min_t = t;
            }
        }
        return min_t;
    }
};

class Creature {
public:
    double pos[2];
    double vel[2];
    double acc[2];
    int finish_time;

    NeuralNet brain;
    std::shared_ptr<DNA> dna;

    double fitness;
    bool crashed;
    bool reached_goal;
    bool is_elite;
    int lifetime;
    double closest_dist;
    
    double min_wall_dist;
    double avg_x;
    
    // color RGB
    int r, g, b;

    std::vector<Point> path_history;

    Creature(Point start_pos, std::shared_ptr<DNA> dna_source, bool is_elite = false);
    
    void apply_force(const std::vector<double>& force);
    std::vector<double> get_sensor_data(const std::vector<Obstacle>& obstacles, Point target_pos);
    void update(int tick, Point target_pos, double width, double height, const std::vector<Obstacle>& obstacles);
    void calc_fitness(Point target_pos);
};

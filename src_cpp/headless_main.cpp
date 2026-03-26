#include "population.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <future>
#include <mutex>
#include <numeric>
#include <sstream>

const int SIM_WIDTH = 800;
const int SIM_HEIGHT = 600;
const int GEN_TTL = 800;
const int POP_SIZE = 800;

struct SimulationResult {
    double elapsed;
    double best_fitness;
    int best_path_points;
    int last_successes;
    int last_crashes;
    double last_avg_fit;
    double last_min_distance;
    double last_avg_life;
    double last_avg_speed;
    double last_dna_diversity;
    double last_success_rate;
    int last_best_finish_time;
    double last_avg_finish_time;
    
    int peak_successes;
    double peak_success_rate;
    int first_success_gen;
    int generations_with_success;
    int fastest_finish_time;
    
    int max_gens;
    unsigned int seed;
    std::string telemetry_path;
    int run_id;
};

struct SimpleRect {
    int x, y, w, h;
    SimpleRect(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {}
};

SimulationResult run_simulation(int max_gens, std::string telemetry_path, unsigned int seed, bool verbose, int run_id) {
    if (seed != 0) {
        srand(seed);
    }

    Point target_pos = {(double)SIM_WIDTH / 2, 50.0};
    Point start_pos = {(double)SIM_WIDTH / 2, (double)SIM_HEIGHT - 50.0};

    std::vector<Obstacle> static_obstacles = {
        {100.0, 400.0, 200.0, 80.0},
        {500.0, 400.0, 200.0, 80.0},
        {250.0, 250.0, 300.0, 80.0},
        {50.0, 100.0, 200.0, 50.0},
        {550.0, 100.0, 200.0, 50.0}
    };

    Population pop(POP_SIZE, 0.02, start_pos, target_pos);

    int frame_count = 0;
    int gen = 1;

    double best_fitness_ever = 0.0;
    int stagnation = 0;
    
    SimulationResult res = {};
    res.max_gens = max_gens;
    res.seed = seed;
    res.telemetry_path = telemetry_path;
    res.run_id = run_id;
    res.last_min_distance = 999999.0;
    res.last_best_finish_time = -1;
    res.last_avg_finish_time = -1.0;
    res.first_success_gen = -1;
    res.fastest_finish_time = -1;

    std::ofstream telemetry_file;
    if (!telemetry_path.empty()) {
        telemetry_file.open(telemetry_path);
        if (telemetry_file.is_open()) {
            telemetry_file << "Generation,Max_Fitness,Avg_Fitness,Successes,Success_Rate,Crashes,Left_Faction,Right_Faction,Mutation_Rate,Min_Distance,Avg_Lifetime,Avg_Speed,DNA_Diversity,Best_Finish_Time,Avg_Finish_Time\n";
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "Starting headless simulation for " << max_gens << " generations...\n";
    }

    while (gen <= max_gens) {
        int door_y = 100 + static_cast<int>(std::sin(frame_count * 0.05) * 40);
        std::vector<Obstacle> current_obstacles = static_obstacles;
        current_obstacles.push_back({300.0, (double)door_y, 20.0, 100.0});
        current_obstacles.push_back({475.0, (double)door_y, 20.0, 100.0});

        pop.update(frame_count, SIM_WIDTH, SIM_HEIGHT, current_obstacles);
        frame_count++;

        bool all_done = true;
        for (const auto& c : pop.creatures) {
            if (!c->crashed && !c->reached_goal) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            frame_count = GEN_TTL;
        }

        if (frame_count >= GEN_TTL) {
            pop.evaluate_fitness();
            
            res.last_successes = 0;
            res.last_crashes = 0;
            double fitness_total = 0.0;
            double max_fit = -999999.0;
            double min_dist = 999999.0;
            double life_total = 0.0;
            double speed_total = 0.0;
            int right_count = 0;
            int left_count = 0;
            
            int best_finish = 999999;
            double finish_total = 0.0;
            int finish_count = 0;

            for (const auto& c : pop.creatures) {
                if (c->reached_goal) res.last_successes++;
                if (c->crashed) res.last_crashes++;
                
                fitness_total += c->fitness;
                if (c->fitness > max_fit) max_fit = c->fitness;
                
                if (c->closest_dist < min_dist) min_dist = c->closest_dist;
                life_total += c->lifetime;
                speed_total += std::hypot(c->vel[0], c->vel[1]);
                
                if (c->avg_x < SIM_WIDTH/2) left_count++; else right_count++;
                
                if (c->reached_goal) {
                    if (c->finish_time < best_finish) best_finish = c->finish_time;
                    finish_total += c->finish_time;
                    finish_count++;
                }
            }
            
            res.last_avg_fit = fitness_total / pop.size;
            res.last_success_rate = static_cast<double>(res.last_successes) / pop.size;
            res.last_min_distance = min_dist;
            res.last_avg_life = life_total / pop.size;
            res.last_avg_speed = speed_total / pop.size;
            res.last_dna_diversity = 0.0; // Skipping exact variance calculation
            
            if (finish_count > 0) {
                res.last_best_finish_time = best_finish;
                res.last_avg_finish_time = finish_total / finish_count;
            } else {
                res.last_best_finish_time = -1;
                res.last_avg_finish_time = -1.0;
            }
            
            if (res.last_successes > res.peak_successes) res.peak_successes = res.last_successes;
            if (res.last_success_rate > res.peak_success_rate) res.peak_success_rate = res.last_success_rate;
            if (res.last_successes > 0) {
                res.generations_with_success++;
                if (res.first_success_gen == -1) res.first_success_gen = gen;
            }
            if (finish_count > 0) {
                if (res.fastest_finish_time == -1 || best_finish < res.fastest_finish_time) {
                    res.fastest_finish_time = best_finish;
                }
            }

            if (telemetry_file.is_open()) {
                telemetry_file << gen << ","
                               << max_fit << ","
                               << res.last_avg_fit << ","
                               << res.last_successes << ","
                               << res.last_success_rate << ","
                               << res.last_crashes << ","
                               << left_count << ","
                               << right_count << ","
                               << pop.mutation_rate << ","
                               << res.last_min_distance << ","
                               << res.last_avg_life << ","
                               << res.last_avg_speed << ","
                               << res.last_dna_diversity << ",";
                if (finish_count > 0) {
                    telemetry_file << best_finish << "," << res.last_avg_finish_time << "\n";
                } else {
                    telemetry_file << ",\n";
                }
            }

            if (max_fit > best_fitness_ever) {
                best_fitness_ever = max_fit;
                // Skipping path saving
                stagnation = 0;
                pop.mutation_rate = 0.01;
            } else {
                stagnation++;
            }

            if (stagnation == 5) {
                if (verbose) std::cout << "*** Stagnation detected. Spiking mutation! ***\n";
                pop.mutation_rate = 0.03;
            } else if (stagnation > 7) {
                if (verbose) std::cout << "*** Cooling down mutation to stabilize swarm. ***\n";
                pop.mutation_rate = 0.01;
                stagnation = 0;
            }
            
            if (verbose) {
                std::cout << "--- Generation " << gen << " ---\n";
                std::cout << "Success: " << res.last_successes << "/" << pop.size << " | Crashed: " << res.last_crashes << "/" << pop.size << "\n";
                std::cout << "Max Fit: " << max_fit << " | Avg Fit: " << res.last_avg_fit << "\n\n";
            }

            pop.natural_selection();
            frame_count = 0;
            gen++;
        }
    }

    if (telemetry_file.is_open()) {
        telemetry_file.close();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    res.elapsed = std::chrono::duration<double>(end_time - start_time).count();
    res.best_fitness = best_fitness_ever;

    return res;
}

int main(int argc, char** argv) {
    int max_gens = 50;
    int workers = 1;
    int runs = 1;
    bool benchmark = false;
    unsigned int base_seed = 0;
    bool write_telemetry = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--workers" && i + 1 < argc) {
            workers = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            base_seed = std::stoul(argv[++i]);
        } else if (arg == "--telemetry") {
            write_telemetry = true;
        } else {
            try {
                max_gens = std::stoi(arg);
            } catch (...) {}
        }
    }

    std::cout << "Params: max_gens=" << max_gens << " workers=" << workers << " runs=" << runs << "\n";

    auto execute_runs = [&](int w, int r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::future<SimulationResult>> futures;
        std::vector<SimulationResult> results;
        
        for (int run_idx = 0; run_idx < r; ++run_idx) {
            std::string tpath = write_telemetry ? "swarm_telemetry_run_" + std::to_string(run_idx + 1) + ".csv" : "";
            unsigned int seed = base_seed == 0 ? 0 : base_seed + run_idx;
            futures.push_back(std::async(std::launch::async, run_simulation, max_gens, tpath, seed, false, run_idx + 1));
            
            // Wait logic if thread count exceeds max workers
            if (futures.size() >= static_cast<size_t>(w)) {
                bool space_freed = false;
                while (!space_freed) {
                    for (auto it = futures.begin(); it != futures.end(); ) {
                        if (it->wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
                            results.push_back(it->get());
                            it = futures.erase(it);
                            space_freed = true;
                            break; // just free one slot and continue
                        } else {
                            ++it;
                        }
                    }
                }
            }
        }
        
        // Wait for remainder
        for (auto& fut : futures) {
            results.push_back(fut.get());
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "All runs complete | count=" << r << " workers=" << w << " wall_time=" << elapsed << "s throughput=" << (r / elapsed) << " runs/s\n";
    };

    if (benchmark) {
        std::cout << "\n=== Benchmarking Single Worker ===\n";
        execute_runs(1, runs);
        
        std::cout <<("\n=== Benchmarking Multi Worker ===\n");
        execute_runs(workers, runs);
    } else {
        execute_runs(workers, runs);
    }

    return 0;
}

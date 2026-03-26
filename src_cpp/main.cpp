#include "raylib.h"
#include "population.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

const int SIM_WIDTH = 800;
const int SIM_HEIGHT = 600;
const int PANEL_HEIGHT = 240;
const int WIDTH = SIM_WIDTH;
const int HEIGHT = SIM_HEIGHT + PANEL_HEIGHT;
const int GEN_TTL = 800;
const int TELEMETRY_UI_ROWS = 10;

struct Telemetry {
    int gen;
    double max_fit;
    double avg_fit;
    int successes;
    int crashes;
    int left_count;
    int right_count;
    double mut_rate;
};

int main() {
    InitWindow(WIDTH, HEIGHT, "bugBox");
    SetTargetFPS(60);

    Point target_pos = { (double)SIM_WIDTH / 2.0, 50.0 };
    Point start_pos = { (double)SIM_WIDTH / 2.0, SIM_HEIGHT - 50.0 };

    std::vector<Obstacle> static_obstacles = {
        {100, 400, 200, 80},
        {500, 400, 200, 80},
        {250, 250, 300, 80},
        {50, 100, 200, 50},
        {550, 100, 200, 50}
    };
    
    Population pop(800, 0.02, start_pos, target_pos);

    int frame_count = 0;
    int gen = 1;
    double best_fitness_ever = 0.0;
    std::vector<Point> best_path_ever;
    int stagnation = 0;

    std::vector<Telemetry> telemetry_history;

    std::ofstream csv_file("swarm_telemetry.csv");
    if (csv_file.is_open()) {
        csv_file << "Generation,Max_Fitness,Avg_Fitness,Successes,Crashes,Left_Faction,Right_Faction,Mutation_Rate\n";
    }

    int SIM_SPEED = 5;

    Font custom_font = LoadFontEx("../assets/monospace.ttf", 20, 0, 250);

    while (!WindowShouldClose()) {
        for (int step = 0; step < SIM_SPEED; ++step) {
            double door_y = 100.0 + std::sin(frame_count * 0.05) * 40.0;
            std::vector<Obstacle> current_obstacles = static_obstacles;
            current_obstacles.push_back({300, door_y, 20, 100});
            current_obstacles.push_back({475, door_y, 20, 100});

            pop.update(frame_count, SIM_WIDTH, SIM_HEIGHT, current_obstacles);
            frame_count++;

            bool all_done = true;
            for (const auto& c : pop.creatures) {
                if (!c->crashed && !c->reached_goal) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) frame_count = GEN_TTL;

            if (frame_count >= GEN_TTL) {
                pop.evaluate_fitness();
                
                int successes = 0, crashes = 0, left_count = 0;
                double fitness_total = 0.0, max_fit = -1.0;
                std::shared_ptr<Creature> best_bug = nullptr;

                for (const auto& c : pop.creatures) {
                    if (c->reached_goal) successes++;
                    if (c->crashed) crashes++;
                    if (c->avg_x < 400) left_count++;
                    
                    fitness_total += c->fitness;
                    if (c->fitness > max_fit) {
                        max_fit = c->fitness;
                        best_bug = c;
                    }
                }
                
                int right_count = pop.size - left_count;
                double avg_fit = fitness_total / pop.size;

                telemetry_history.push_back({gen, max_fit, avg_fit, successes, crashes, left_count, right_count, pop.mutation_rate});
                if (telemetry_history.size() > TELEMETRY_UI_ROWS) {
                    telemetry_history.erase(telemetry_history.begin());
                }

                if (csv_file.is_open()) {
                    csv_file << gen << "," << max_fit << "," << avg_fit << "," 
                             << successes << "," << crashes << "," << left_count << "," 
                             << right_count << "," << pop.mutation_rate << "\n";
                    csv_file.flush();
                }

                if (max_fit > best_fitness_ever) {
                    best_fitness_ever = max_fit;
                    if (best_bug) best_path_ever = best_bug->path_history;
                    stagnation = 0;
                    pop.mutation_rate = 0.01;
                } else {
                    stagnation++;
                }

                if (stagnation == 5) {
                    pop.mutation_rate = 0.03;
                } else if (stagnation > 7) {
                    pop.mutation_rate = 0.01;
                    stagnation = 0;
                }

                pop.natural_selection();
                frame_count = 0;
                gen++;
            }
        }

        // Draw phase
        BeginDrawing();
        ClearBackground({22, 24, 32, 255});

        if (best_path_ever.size() > 1) {
            for (size_t i = 0; i < best_path_ever.size() - 1; i++) {
                DrawLineEx({(float)best_path_ever[i].x, (float)best_path_ever[i].y},
                           {(float)best_path_ever[i+1].x, (float)best_path_ever[i+1].y}, 
                           3.0f, {233, 193, 112, 255});
            }
        }

        DrawCircle((int)target_pos.x, (int)target_pos.y, 10, {144, 182, 232, 255});

        double door_y = 100.0 + std::sin(frame_count * 0.05) * 40.0;
        std::vector<Obstacle> current_obstacles = static_obstacles;
        current_obstacles.push_back({300, door_y, 20, 100});
        current_obstacles.push_back({475, door_y, 20, 100});

        for (const auto& obs : current_obstacles) {
            DrawRectangle((int)obs.x, (int)obs.y, (int)obs.width, (int)obs.height, {93, 106, 158, 255});
        }

        for (const auto& c : pop.creatures) {
            Color color = {(unsigned char)c->r, (unsigned char)c->g, (unsigned char)c->b, 255};
            if (c->reached_goal) color = {140, 201, 151, 255};
            else if (c->crashed) color = {184, 112, 126, 255};
            
            if (c->is_elite && c->path_history.size() > 1) {
                for (size_t i = 0; i < c->path_history.size() - 1; i++) {
                    DrawLineV({(float)c->path_history[i].x, (float)c->path_history[i].y},
                              {(float)c->path_history[i+1].x, (float)c->path_history[i+1].y}, color);
                }
            }
            DrawCircle((int)c->pos[0], (int)c->pos[1], 5, color);
        }

        DrawTextEx(custom_font, TextFormat("Gen: %d | Frame: %d", gen, frame_count), {10, 10}, 20, 1, {176, 214, 190, 255});

        // Panel
        DrawRectangle(0, SIM_HEIGHT, WIDTH, PANEL_HEIGHT, {14, 16, 22, 255});
        DrawLine(0, SIM_HEIGHT, WIDTH, SIM_HEIGHT, {245, 226, 162, 255});

        int panel_y = SIM_HEIGHT + 20;
        const char* headers[] = {"Gen", "MaxF", "AvgF", "Succ", "Crash", "Left", "Right", "Mut"};
        int col_x[] = {20, 115, 210, 305, 400, 495, 590, 685};
        
        for (int i = 0; i < 8; i++) {
            DrawTextEx(custom_font, headers[i], {(float)col_x[i], (float)panel_y}, 20, 1, {245, 226, 162, 255});
        }

        int row_idx = 1;
        for (auto it = telemetry_history.rbegin(); it != telemetry_history.rend(); ++it) {
            float y = panel_y + row_idx * 24;
            DrawTextEx(custom_font, TextFormat("%d", it->gen), {(float)col_x[0], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%.2f", it->max_fit), {(float)col_x[1], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%.2f", it->avg_fit), {(float)col_x[2], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%d", it->successes), {(float)col_x[3], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%d", it->crashes), {(float)col_x[4], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%d", it->left_count), {(float)col_x[5], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%d", it->right_count), {(float)col_x[6], y}, 20, 1, {238, 244, 252, 255});
            DrawTextEx(custom_font, TextFormat("%.2f", it->mut_rate), {(float)col_x[7], y}, 20, 1, {238, 244, 252, 255});
            row_idx++;
        }

        EndDrawing();
    }
    UnloadFont(custom_font);
    CloseWindow();
    return 0;
}

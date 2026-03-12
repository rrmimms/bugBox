import re

with open("/home/robert/personalProjects/bugBox/src_cpp/headless_main.cpp", "r") as f:
    code = f.read()

# Fix types
code = code.replace("std::vector<int> target_pos = {SIM_WIDTH / 2, 50};", "Point target_pos = {(double)SIM_WIDTH / 2, 50.0};")
code = code.replace("std::vector<int> start_pos = {SIM_WIDTH / 2, SIM_HEIGHT - 50};", "Point start_pos = {(double)SIM_WIDTH / 2, (double)SIM_HEIGHT - 50.0};")

code = code.replace("std::vector<std::vector<int>> static_obstacles = {", "std::vector<Obstacle> static_obstacles = {")
code = code.replace("{100, 400, 200, 80}", "{100.0, 400.0, 200.0, 80.0}")
code = code.replace("{500, 400, 200, 80}", "{500.0, 400.0, 200.0, 80.0}")
code = code.replace("{250, 250, 300, 80}", "{250.0, 250.0, 300.0, 80.0}")
code = code.replace("{50, 100, 200, 50}", "{50.0, 100.0, 200.0, 50.0}")
code = code.replace("{550, 100, 200, 50}", "{550.0, 100.0, 200.0, 50.0}")

code = code.replace("Population pop(POP_SIZE, 0.02, start_pos, target_pos, GEN_TTL);", "Population pop(POP_SIZE, 0.02, start_pos, target_pos);")

code = code.replace("std::vector<std::vector<int>> current_obstacles = static_obstacles;", "std::vector<Obstacle> current_obstacles = static_obstacles;")
code = code.replace("current_obstacles.push_back({300, door_y, 20, 100});", "current_obstacles.push_back({300.0, (double)door_y, 20.0, 100.0});")
code = code.replace("current_obstacles.push_back({475, door_y, 20, 100});", "current_obstacles.push_back({475.0, (double)door_y, 20.0, 100.0});")

# Fix pointer accesses
code = code.replace("c.crashed", "c->crashed")
code = code.replace("c.reached_goal", "c->reached_goal")
code = code.replace("c.fitness", "c->fitness")
code = code.replace("c.closest_dist", "c->closest_dist")
code = code.replace("c.lifetime", "c->lifetime")
code = code.replace("c.vel[0]", "c->vel[0]")
code = code.replace("c.vel[1]", "c->vel[1]")
code = code.replace("c.avg_x", "c->avg_x")
code = code.replace("c.finish_time", "c->finish_time")

with open("/home/robert/personalProjects/bugBox/src_cpp/headless_main.cpp", "w") as f:
    f.write(code)

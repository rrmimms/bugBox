cmake_minimum_required(VERSION 3.14)
project(BugBox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/5.0.tar.gz
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

include_directories(include)

add_executable(bugbox
    src_cpp/main.cpp
    src_cpp/dna.cpp
    src_cpp/nn/nn.cpp
    src_cpp/creature.cpp
    src_cpp/population.cpp
)

target_link_libraries(bugbox PRIVATE raylib)

add_executable(bugbox_headless
    src_cpp/headless_main.cpp
    src_cpp/dna.cpp
    src_cpp/nn/nn.cpp
    src_cpp/creature.cpp
    src_cpp/population.cpp
)
target_link_libraries(bugbox_headless PRIVATE pthread)

cmake_minimum_required(VERSION 3.18)
project(FlockingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Включаем поддержку CUDA
enable_language(CUDA)

add_executable(flocking_sim
    main.cpp
    renderer.cpp
    simulation.cu
)

set_target_properties(flocking_sim PROPERTIES
    CUDA_ARCHITECTURES "89"   # для RTX 4080 (Compute Capability 8.9, можно указать 86)
)

target_include_directories(flocking_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flocking_sim OpenGL::GL glfw ${CUDA_LIBRARIES})
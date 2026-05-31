cmake_minimum_required(VERSION 3.18)
project(FlockingCPUOnly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Поиск зависимостей
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)

# CPU‑версия симуляции (без CUDA)
add_executable(flocking_cpu
    main.cpp
    renderer.cpp
    simulation_cpu.cpp
)

# Определяем макрос, чтобы main.cpp использовал CPU-класс
target_compile_definitions(flocking_cpu PRIVATE USE_CPU_SIMULATION)

target_include_directories(flocking_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(flocking_cpu
    OpenGL::GL
    glfw
    GLEW::GLEW
)
cmake_minimum_required(VERSION 3.18)
project(FlockingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Архитектура CUDA (RTX 4080 = Compute Capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# --- Поиск зависимостей ---
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)   # vcpkg предоставляет glfw3Config.cmake
find_package(GLEW REQUIRED)           # vcpkg предоставляет GLEW

# --- Основной исполняемый файл ---
add_executable(flocking_sim
    main.cpp
    renderer.cpp
    simulation.cu
)

# Подключаем заголовки
target_include_directories(flocking_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Линкуем библиотеки
target_link_libraries(flocking_sim
    OpenGL::GL
    glfw
    GLEW::GLEW
    ${CUDA_LIBRARIES}
)

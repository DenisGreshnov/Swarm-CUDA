#pragma once
#include <GL/glew.h>            // ← ВСЕГДА ПЕРВЫМ
#include <GLFW/glfw3.h>
#include "simulation.cuh"
#include <string>
#include <chrono>

class Renderer {
private:
    GLFWwindow* window;
    int window_width, window_height;

    // Камера
    Vector2 camera_offset{0.0, 0.0};
    float zoom = 1.0f;
    bool panning = false;
    double last_mouse_x = 0.0, last_mouse_y = 0.0;

    // Измерение времени
    std::chrono::steady_clock::time_point last_frame_time;
    int frame_count = 0;
    double fps = 0.0;
    double frame_time_ms = 0.0;
    double sim_time_ms = 0.0;

    // Шейдерная программа и буферы
    unsigned int shader_program = 0;
    unsigned int vao_agents = 0, vbo_agents = 0;
    unsigned int vao_obstacles = 0, vbo_obstacles = 0;
    unsigned int vao_beta = 0, vbo_beta = 0;
    unsigned int vao_target = 0, vbo_target = 0;
    unsigned int vao_connections = 0, vbo_connections = 0;
    unsigned int vao_grid = 0, vbo_grid = 0;

    // Функции для построения геометрии
    void build_agents_geometry(const std::vector<Agent>& agents);
    void build_obstacles_geometry(const std::vector<Obstacle>& obstacles);
    void build_beta_geometry(const std::vector<BetaAgent>& beta_agents);
    void build_target_geometry(const Vector2& target, bool enabled);
    void build_connections_geometry(const FlockSimulation& simulation);
    void build_grid_geometry();

    // Загрузка и компиляция шейдеров
    unsigned int load_shaders(const char* vertex_source, const char* fragment_source);

public:
    Renderer(int width = 1000, int height = 800);
    ~Renderer();

    bool initialize();
    void render(FlockSimulation& simulation);
    bool should_close() const;
    void poll_events();

    // Установка callback'ов мыши и клавиатуры
    void setup_callbacks(FlockSimulation* sim);

    // Получение состояния
    GLFWwindow* get_window() const { return window; }
    int get_window_width() const { return window_width; }
    int get_window_height() const { return window_height; }

    // Преобразование координат с учётом камеры
    Vector2 screen_to_world(double screen_x, double screen_y) const;

    // Установка времени симуляции для отображения
    void set_sim_time(double ms) { sim_time_ms = ms; }

    // Обработчики ввода (вызываются из статических callback'ов)
    void on_mouse_button(int button, int action, int mods);
    void on_cursor_pos(double xpos, double ypos);
    void on_scroll(double xoffset, double yoffset);
    void on_key(int key, int action, int mods);
};
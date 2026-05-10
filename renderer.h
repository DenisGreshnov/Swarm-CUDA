#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "simulation.cuh"
#include <string>
#include <chrono>

class Renderer {
private:
    GLFWwindow* window;
    int window_width, window_height;

    Vector2 camera_offset{0.0f, 0.0f};
    float zoom = 1.0f;
    bool panning = false;
    double last_mouse_x = 0.0, last_mouse_y = 0.0;

    std::chrono::steady_clock::time_point last_frame_time;
    int frame_count = 0;
    double fps = 0.0;
    double frame_time_ms = 0.0;
    double sim_time_ms = 0.0;

    unsigned int shader_program = 0;

    unsigned int vao_agents = 0, vbo_agents = 0;
    unsigned int vao_obstacles = 0, vbo_obstacles = 0;
    unsigned int vao_beta = 0, vbo_beta = 0;
    unsigned int vao_target = 0, vbo_target = 0;
    unsigned int vao_connections = 0, vbo_connections = 0;
    unsigned int vao_grid = 0, vbo_grid = 0;

    int grid_vertex_count = 0;

    bool obstacles_dirty = true;
    bool target_dirty = true;

    void build_obstacles_geometry(const std::vector<Obstacle>& obstacles);
    void build_target_geometry(const Vector2& target, bool enabled);
    void build_grid_geometry();

    unsigned int load_shaders(const char* vertex_source, const char* fragment_source);
    void get_visible_bounds(float& left, float& right, float& bottom, float& top) const;

public:
    Renderer(int width = 1000, int height = 800);
    ~Renderer();

    bool initialize(FlockSimulation& simulation);
    void render(FlockSimulation& simulation);
    bool should_close() const;
    void poll_events();

    void setup_callbacks(FlockSimulation* sim);
    GLFWwindow* get_window() const { return window; }
    void update_window_size(int width, int height) {
        window_width = width;
        window_height = height;
        glViewport(0, 0, width, height);
    }

    Vector2 screen_to_world(double screen_x, double screen_y) const;
    void set_sim_time(double ms) { sim_time_ms = ms; }

    void mark_obstacles_dirty() { obstacles_dirty = true; }
    void mark_target_dirty()    { target_dirty = true; }

    void on_mouse_button(int button, int action, int mods);
    void on_cursor_pos(double xpos, double ypos);
    void on_scroll(double xoffset, double yoffset);
    void on_key(int key, int action, int mods);
};
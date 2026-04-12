#pragma once
#include "simulation.cuh"
#include <GLFW/glfw3.h>

class Renderer {
private:
    GLFWwindow* window;
    int window_width, window_height;
    
public:
    Renderer(int width = 800, int height = 600);
    ~Renderer();
    
    bool initialize();
    void render(FlockSimulation& simulation);
    bool should_close() const;
    void poll_events();
    
    GLFWwindow* get_window() const { return window; }
    Vector2 screen_to_world(double screen_x, double screen_y) const;
    int get_window_width() const { return window_width; }
    int get_window_height() const { return window_height; }
    
private:
    void draw_agent(const Agent& agent);
    void draw_obstacle(const Obstacle& obstacle);
    void draw_beta_agent(const BetaAgent& beta_agent);
    void draw_target(const Vector2& target);
    void draw_connections(const FlockSimulation& simulation); // НОВОЕ: отрисовка сетки связей
};
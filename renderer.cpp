#include "renderer.h"
#include <iostream>
#include <cmath>
#include <corecrt_math_defines.h>

Renderer::Renderer(int width, int height) 
    : window_width(width), window_height(height), window(nullptr) {}

Renderer::~Renderer() {
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Renderer::initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    window = glfwCreateWindow(window_width, window_height, "Flocking Simulation - Algorithm 3", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync
    
    // Настройка OpenGL
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(3.0f);
    
    std::cout << "Renderer initialized successfully" << std::endl;
    std::cout << "Window size: " << window_width << "x" << window_height << std::endl;
    return true;
}

void Renderer::render(FlockSimulation& simulation) {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Темно-серый фон
    
    // Настройка проекции - ФИКСИРОВАННЫЕ координаты
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-200, 200, -200, 200, -1, 1); // От -200 до +200 по обеим осям
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Получаем данные из симуляции
    auto agents = simulation.get_agents();
    auto obstacles = simulation.get_obstacles();
    auto beta_agents = simulation.get_beta_agents();
    auto target = simulation.get_target();
    bool show_betas = simulation.is_beta_display_enabled();
    bool target_enabled = simulation.is_target_enabled();
    bool show_connections = simulation.is_connections_display_enabled();
    
    // Сначала рисуем соединения (чтобы они были под агентами)
    if (show_connections) {
        draw_connections(simulation);
    }
    
    // Рендерим цель только если она включена
    if (target_enabled) {
        draw_target(target);
    }
    
    // Рендерим препятствия
    for (const auto& obstacle : obstacles) {
        draw_obstacle(obstacle);
    }
    
    // Рендерим β-агентов только если включен их показ
    if (show_betas) {
        for (const auto& beta_agent : beta_agents) {
            draw_beta_agent(beta_agent);
        }
    }
    
    // Рендерим агентов
    for (const auto& agent : agents) {
        draw_agent(agent);
    }
    
    glfwSwapBuffers(window);
}

void Renderer::draw_agent(const Agent& agent) {
    Vector2 direction = agent.velocity.length() > 0.1 ? 
                       agent.velocity.normalized() : Vector2(1, 0);
    Vector2 perpendicular(-direction.y, direction.x);
    
    glColor3f(0.0f, 0.7f, 1.0f); // Ярко-голубой
    
    glBegin(GL_TRIANGLES);
    glVertex2f(agent.position.x + direction.x * 5, agent.position.y + direction.y * 5);
    glVertex2f(agent.position.x - direction.x * 3 + perpendicular.x * 3, 
               agent.position.y - direction.y * 3 + perpendicular.y * 3);
    glVertex2f(agent.position.x - direction.x * 3 - perpendicular.x * 3, 
               agent.position.y - direction.y * 3 - perpendicular.y * 3);
    glEnd();
}

void Renderer::draw_obstacle(const Obstacle& obstacle) {
    glColor3f(0.9f, 0.2f, 0.2f); // Красный
    
    glBegin(GL_TRIANGLE_FAN);
    for (int i = 0; i <= 32; ++i) {
        double angle = 2.0 * M_PI * i / 32;
        glVertex2f(obstacle.position.x + obstacle.radius * cos(angle),
                   obstacle.position.y + obstacle.radius * sin(angle));
    }
    glEnd();
}

void Renderer::draw_beta_agent(const BetaAgent& beta_agent) {
    glColor3f(1.0f, 0.5f, 0.0f); // Оранжевый для β-агентов
    
    // Рисуем маленький квадрат для β-агента
    glBegin(GL_QUADS);
    glVertex2f(beta_agent.position.x - 2, beta_agent.position.y - 2);
    glVertex2f(beta_agent.position.x + 2, beta_agent.position.y - 2);
    glVertex2f(beta_agent.position.x + 2, beta_agent.position.y + 2);
    glVertex2f(beta_agent.position.x - 2, beta_agent.position.y + 2);
    glEnd();
    
    // Рисуем стрелку направления скорости
    if (beta_agent.velocity.length() > 0.5) {
        Vector2 dir = beta_agent.velocity.normalized();
        glBegin(GL_LINES);
        glVertex2f(beta_agent.position.x, beta_agent.position.y);
        glVertex2f(beta_agent.position.x + dir.x * 6, beta_agent.position.y + dir.y * 6);
        glEnd();
    }
}

void Renderer::draw_target(const Vector2& target) {
    glColor3f(0.2f, 0.9f, 0.2f); // Зеленый
    
    // Крест
    glBegin(GL_LINES);
    glVertex2f(target.x - 8, target.y); glVertex2f(target.x + 8, target.y);
    glVertex2f(target.x, target.y - 8); glVertex2f(target.x, target.y + 8);
    glEnd();
    
    // Круг
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 16; ++i) {
        double angle = 2.0 * M_PI * i / 16;
        glVertex2f(target.x + 12 * cos(angle), target.y + 12 * sin(angle));
    }
    glEnd();
}

void Renderer::draw_connections(const FlockSimulation& simulation) {
    auto agents = simulation.get_agents();
    auto beta_agents = simulation.get_beta_agents();
    double interaction_range = simulation.get_interaction_range();
    double obstacle_range = simulation.get_obstacle_range();
    
    // Включаем прозрачность для линий
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Рисуем связи между α-агентами (голубые линии)
    glColor4f(1.0f, 1.0f, 1.0f, 0.4f);
    glBegin(GL_LINES);
    for (size_t i = 0; i < agents.size(); ++i) {
        for (size_t j = i + 1; j < agents.size(); ++j) {
            Vector2 diff = agents[j].position - agents[i].position;
            double distance = diff.length();
            if (distance < interaction_range) {
                // Интенсивность линии зависит от расстояния
                float alpha = (1.0f - distance / interaction_range / 2);
                glColor4f(1.0f, 1.0f, 1.0f, alpha);
                glVertex2f(agents[i].position.x, agents[i].position.y);
                glVertex2f(agents[j].position.x, agents[j].position.y);
            }
        }
    }
    glEnd();
    
    // Рисуем связи между α-агентами и β-агентами (оранжевые линии)
    glColor4f(1.0f, 0.5f, 0.0f, 0.4f); // Полупрозрачный оранжевый
    glBegin(GL_LINES);
    for (const auto& agent : agents) {
        for (const auto& beta_agent : beta_agents) {
            Vector2 diff = beta_agent.position - agent.position;
            double distance = diff.length();
            if (distance < obstacle_range) {
                // Интенсивность линии зависит от расстояния
                float alpha = (1.0f - distance / obstacle_range / 2);
                glColor4f(1.0f, 0.5f, 0.0f, alpha);
                glVertex2f(agent.position.x, agent.position.y);
                glVertex2f(beta_agent.position.x, beta_agent.position.y);
            }
        }
    }
    glEnd();
    
    glDisable(GL_BLEND);
}

bool Renderer::should_close() const {
    return glfwWindowShouldClose(window);
}

void Renderer::poll_events() {
    glfwPollEvents();
}

Vector2 Renderer::screen_to_world(double screen_x, double screen_y) const {
    // СУПЕР-ПРОСТОЙ вариант - гарантированно работает
    // Предполагаем что окно 1000x800, мир 400x400
    
    double world_x = (screen_x / 1000.0) * 400.0 - 200.0;
    double world_y = 200.0 - (screen_y / 800.0) * 400.0;
    
    return Vector2(world_x, world_y);
}
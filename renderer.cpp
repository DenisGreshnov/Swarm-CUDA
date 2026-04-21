#include "renderer.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <sstream>   // для форматирования времени
#include <iomanip>   // для std::setprecision
#include <corecrt_math_defines.h>

// ----------------------------------------------------------------------
// Шейдеры (встроенные в код)
// ----------------------------------------------------------------------
const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uProjection;
uniform mat4 uView;

out vec3 vColor;

void main() {
    gl_Position = uProjection * uView * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
)";

const char* fragment_shader_source = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

// ----------------------------------------------------------------------
// Вспомогательные функции для построения геометрии
// ----------------------------------------------------------------------
struct Vertex {
    float x, y;
    float r, g, b;
};

static void add_triangle(std::vector<Vertex>& vertices, Vector2 p1, Vector2 p2, Vector2 p3, float r, float g, float b) {
    vertices.push_back({(float)p1.x, (float)p1.y, r, g, b});
    vertices.push_back({(float)p2.x, (float)p2.y, r, g, b});
    vertices.push_back({(float)p3.x, (float)p3.y, r, g, b});
}

static void add_circle(std::vector<Vertex>& vertices, Vector2 center, float radius, int segments, float r, float g, float b) {
    for (int i = 0; i < segments; ++i) {
        double angle1 = 2.0 * M_PI * i / segments;
        double angle2 = 2.0 * M_PI * (i + 1) / segments;
        Vector2 p1 = center + Vector2(radius * cos(angle1), radius * sin(angle1));
        Vector2 p2 = center + Vector2(radius * cos(angle2), radius * sin(angle2));
        add_triangle(vertices, center, p1, p2, r, g, b);
    }
}

static void add_line(std::vector<Vertex>& vertices, Vector2 p1, Vector2 p2, float r, float g, float b) {
    vertices.push_back({(float)p1.x, (float)p1.y, r, g, b});
    vertices.push_back({(float)p2.x, (float)p2.y, r, g, b});
}

// ----------------------------------------------------------------------
// Renderer implementation
// ----------------------------------------------------------------------
Renderer::Renderer(int width, int height)
    : window_width(width), window_height(height), window(nullptr) {}

Renderer::~Renderer() {
    if (window) {
        glDeleteProgram(shader_program);
        glDeleteVertexArrays(1, &vao_agents);
        glDeleteBuffers(1, &vbo_agents);
        glDeleteVertexArrays(1, &vao_obstacles);
        glDeleteBuffers(1, &vbo_obstacles);
        glDeleteVertexArrays(1, &vao_beta);
        glDeleteBuffers(1, &vbo_beta);
        glDeleteVertexArrays(1, &vao_target);
        glDeleteBuffers(1, &vbo_target);
        glDeleteVertexArrays(1, &vao_connections);
        glDeleteBuffers(1, &vbo_connections);
        glDeleteVertexArrays(1, &vao_grid);
        glDeleteBuffers(1, &vbo_grid);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Renderer::initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(window_width, window_height, "Flocking Simulation - GPU Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync
    glewExperimental = GL_TRUE;  // Для Core Profile
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    // Загрузка шейдеров
    shader_program = load_shaders(vertex_shader_source, fragment_shader_source);
    if (!shader_program) return false;

    // Создание VAO для разных объектов
    glGenVertexArrays(1, &vao_agents);
    glGenBuffers(1, &vbo_agents);
    glGenVertexArrays(1, &vao_obstacles);
    glGenBuffers(1, &vbo_obstacles);
    glGenVertexArrays(1, &vao_beta);
    glGenBuffers(1, &vbo_beta);
    glGenVertexArrays(1, &vao_target);
    glGenBuffers(1, &vbo_target);
    glGenVertexArrays(1, &vao_connections);
    glGenBuffers(1, &vbo_connections);
    glGenVertexArrays(1, &vao_grid);
    glGenBuffers(1, &vbo_grid);

    // Построение статической сетки
    build_grid_geometry();

    // Начальное время для измерения FPS
    last_frame_time = std::chrono::steady_clock::now();

    std::cout << "Renderer initialized successfully (Modern OpenGL)" << std::endl;
    return true;
}

unsigned int Renderer::load_shaders(const char* vertex_source, const char* fragment_source) {
    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertex_source, nullptr);
    glCompileShader(vertex);
    int success;
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(vertex, 512, nullptr, info);
        std::cerr << "Vertex shader compilation failed: " << info << std::endl;
        return 0;
    }

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragment_source, nullptr);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(fragment, 512, nullptr, info);
        std::cerr << "Fragment shader compilation failed: " << info << std::endl;
        return 0;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info[512];
        glGetProgramInfoLog(program, 512, nullptr, info);
        std::cerr << "Shader linking failed: " << info << std::endl;
        return 0;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return program;
}

void Renderer::render(FlockSimulation& simulation) {
    // Измерение времени кадра
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - last_frame_time;
    frame_time_ms = elapsed.count() * 1000.0;
    last_frame_time = now;

    frame_count++;
    static auto last_fps_update = now;
    if (std::chrono::duration<double>(now - last_fps_update).count() >= 1.0) {
        fps = frame_count;
        frame_count = 0;
        last_fps_update = now;

        // Обновление заголовка окна
        std::ostringstream title_ss;
        title_ss << "Flocking | FPS: " << (int)fps
                 << " | Frame: " << std::fixed << std::setprecision(2) << frame_time_ms << " ms"
                 << " | Sim: " << sim_time_ms << " ms";
        std::string title = title_ss.str();
        glfwSetWindowTitle(window, title.c_str());
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Включение прозрачности для соединений
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Установка матриц проекции и вида с учётом камеры
    float left   = -200.0f / zoom + camera_offset.x;
    float right  =  200.0f / zoom + camera_offset.x;
    float bottom = -200.0f / zoom + camera_offset.y;
    float top    =  200.0f / zoom + camera_offset.y;

    float projection[16] = {
        2.0f/(right-left), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f/(top-bottom), 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -(right+left)/(right-left), -(top+bottom)/(top-bottom), 0.0f, 1.0f
    };
    float view[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    glUseProgram(shader_program);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "uProjection"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "uView"), 1, GL_FALSE, view);

    // Получение данных симуляции
    auto agents = simulation.get_agents();
    auto obstacles = simulation.get_obstacles();
    auto beta_agents = simulation.get_beta_agents();
    auto target = simulation.get_target();
    bool show_betas = simulation.is_beta_display_enabled();
    bool target_enabled = simulation.is_target_enabled();
    bool show_connections = simulation.is_connections_display_enabled();

    // Построение геометрии (динамически каждый кадр – для простоты)
    build_agents_geometry(agents);
    build_obstacles_geometry(obstacles);
    if (show_betas) build_beta_geometry(beta_agents);
    build_target_geometry(target, target_enabled);
    if (show_connections) build_connections_geometry(simulation);

    // Рисование сетки
    glBindVertexArray(vao_grid);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid);
    int grid_vert_count;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &grid_vert_count);
    glDrawArrays(GL_LINES, 0, grid_vert_count / sizeof(Vertex));

    // Рисование соединений (с прозрачностью)
    if (show_connections && vbo_connections) {
        glBindVertexArray(vao_connections);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_connections);
        int count;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &count);
        glDrawArrays(GL_LINES, 0, count / sizeof(Vertex));
    }

    // Рисование препятствий
    glBindVertexArray(vao_obstacles);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacles);
    int obs_vert_count;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &obs_vert_count);
    glDrawArrays(GL_TRIANGLES, 0, obs_vert_count / sizeof(Vertex));

    // Рисование цели
    if (target_enabled) {
        glBindVertexArray(vao_target);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
        int target_vert_count;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &target_vert_count);
        glDrawArrays(GL_LINES, 0, target_vert_count / sizeof(Vertex));
    }

    // Рисование β-агентов
    if (show_betas && !beta_agents.empty()) {
        glBindVertexArray(vao_beta);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_beta);
        int beta_vert_count;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &beta_vert_count);
        glDrawArrays(GL_TRIANGLES, 0, beta_vert_count / sizeof(Vertex));
    }

    // Рисование агентов (поверх всего)
    glBindVertexArray(vao_agents);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_agents);
    int agent_vert_count;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &agent_vert_count);
    glDrawArrays(GL_TRIANGLES, 0, agent_vert_count / sizeof(Vertex));

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);

    glfwSwapBuffers(window);
}

// ----------------------------------------------------------------------
// Построение геометрии в VBO
// ----------------------------------------------------------------------
void Renderer::build_agents_geometry(const std::vector<Agent>& agents) {
    std::vector<Vertex> vertices;
    for (const auto& agent : agents) {
        Vector2 dir = agent.velocity.length() > 0.1 ? agent.velocity.normalized() : Vector2(1, 0);
        Vector2 perp(-dir.y, dir.x);
        Vector2 tip = agent.position + dir * 5;
        Vector2 left = agent.position - dir * 3 + perp * 3;
        Vector2 right = agent.position - dir * 3 - perp * 3;
        add_triangle(vertices, tip, left, right, 0.0f, 0.7f, 1.0f);
    }

    glBindVertexArray(vao_agents);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_agents);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_obstacles_geometry(const std::vector<Obstacle>& obstacles) {
    std::vector<Vertex> vertices;
    for (const auto& obs : obstacles) {
        add_circle(vertices, obs.position, (float)obs.radius, 32, 0.9f, 0.2f, 0.2f);
    }
    glBindVertexArray(vao_obstacles);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacles);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_beta_geometry(const std::vector<BetaAgent>& beta_agents) {
    std::vector<Vertex> vertices;
    for (const auto& beta : beta_agents) {
        // Маленький квадрат
        Vector2 p1 = beta.position + Vector2(-2, -2);
        Vector2 p2 = beta.position + Vector2( 2, -2);
        Vector2 p3 = beta.position + Vector2( 2,  2);
        Vector2 p4 = beta.position + Vector2(-2,  2);
        add_triangle(vertices, p1, p2, p3, 1.0f, 0.5f, 0.0f);
        add_triangle(vertices, p1, p3, p4, 1.0f, 0.5f, 0.0f);
        // Линия направления
        if (beta.velocity.length() > 0.5) {
            Vector2 dir = beta.velocity.normalized();
            Vector2 end = beta.position + dir * 6;
            add_line(vertices, beta.position, end, 1.0f, 0.5f, 0.0f);
        }
    }
    glBindVertexArray(vao_beta);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_beta);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_target_geometry(const Vector2& target, bool enabled) {
    std::vector<Vertex> vertices;
    if (!enabled) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        return;
    }

    // Крест
    add_line(vertices, target + Vector2(-8, 0), target + Vector2(8, 0), 0.2f, 0.9f, 0.2f);
    add_line(vertices, target + Vector2(0, -8), target + Vector2(0, 8), 0.2f, 0.9f, 0.2f);
    // Окружность (линиями)
    const int seg = 16;
    for (int i = 0; i < seg; ++i) {
        double a1 = 2.0 * M_PI * i / seg;
        double a2 = 2.0 * M_PI * (i+1) / seg;
        Vector2 p1 = target + Vector2(12 * cos(a1), 12 * sin(a1));
        Vector2 p2 = target + Vector2(12 * cos(a2), 12 * sin(a2));
        add_line(vertices, p1, p2, 0.2f, 0.9f, 0.2f);
    }

    glBindVertexArray(vao_target);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_connections_geometry(const FlockSimulation& simulation) {
    std::vector<Vertex> vertices;
    auto agents = simulation.get_agents();
    auto beta_agents = simulation.get_beta_agents();
    double alpha_range = simulation.get_interaction_range();
    double beta_range = simulation.get_obstacle_range();

    // α-α связи (белые полупрозрачные)
    for (size_t i = 0; i < agents.size(); ++i) {
        for (size_t j = i+1; j < agents.size(); ++j) {
            Vector2 diff = agents[j].position - agents[i].position;
            double dist = diff.length();
            if (dist < alpha_range) {
                float alpha = 1.0f - (float)(dist / alpha_range);
                add_line(vertices, agents[i].position, agents[j].position, 1.0f, 1.0f, 1.0f);
                // Прозрачность задаётся через uniform, но здесь для простоты цвет постоянный.
                // В реальном приложении лучше вынести в отдельный uniform.
            }
        }
    }

    // α-β связи (оранжевые)
    for (const auto& agent : agents) {
        for (const auto& beta : beta_agents) {
            Vector2 diff = beta.position - agent.position;
            double dist = diff.length();
            if (dist < beta_range) {
                add_line(vertices, agent.position, beta.position, 1.0f, 0.5f, 0.0f);
            }
        }
    }

    glBindVertexArray(vao_connections);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_connections);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_grid_geometry() {
    std::vector<Vertex> vertices;
    float step = 20.0f;
    float range = 200.0f;
    // Вертикальные линии
    for (float x = -range; x <= range; x += step) {
        add_line(vertices, Vector2(x, -range), Vector2(x, range), 0.3f, 0.3f, 0.3f);
    }
    // Горизонтальные линии
    for (float y = -range; y <= range; y += step) {
        add_line(vertices, Vector2(-range, y), Vector2(range, y), 0.3f, 0.3f, 0.3f);
    }

    glBindVertexArray(vao_grid);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

// ----------------------------------------------------------------------
// Обработка ввода (камера)
// ----------------------------------------------------------------------
void Renderer::setup_callbacks(FlockSimulation* sim) {
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
        Renderer* r = static_cast<Renderer*>(glfwGetWindowUserPointer(w));
        r->on_mouse_button(button, action, mods);
        // Также вызываем старый колбэк для установки цели/препятствий
        // (нужно сохранить исходный указатель на симуляцию)
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        Renderer* r = static_cast<Renderer*>(glfwGetWindowUserPointer(w));
        r->on_cursor_pos(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double xoff, double yoff) {
        Renderer* r = static_cast<Renderer*>(glfwGetWindowUserPointer(w));
        r->on_scroll(xoff, yoff);
    });
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
        Renderer* r = static_cast<Renderer*>(glfwGetWindowUserPointer(w));
        r->on_key(key, action, mods);
    });
}

void Renderer::on_mouse_button(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            panning = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            panning = false;
        }
    }
    // Левую кнопку обрабатываем в main.cpp, но можно и здесь.
}

void Renderer::on_cursor_pos(double xpos, double ypos) {
    if (panning) {
        double dx = xpos - last_mouse_x;
        double dy = ypos - last_mouse_y;
        // Масштабирование с учётом зума и размера окна
        float world_dx = (dx / window_width) * (400.0f / zoom);
        float world_dy = (dy / window_height) * (400.0f / zoom);
        camera_offset.x -= world_dx;
        camera_offset.y += world_dy; // ось Y перевёрнута
        last_mouse_x = xpos;
        last_mouse_y = ypos;
    }
}

void Renderer::on_scroll(double xoffset, double yoffset) {
    zoom *= (1.0f + (float)yoffset * 0.1f);
    zoom = std::max(0.2f, std::min(zoom, 5.0f));
}

void Renderer::on_key(int key, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        float pan_speed = 10.0f / zoom;
        switch (key) {
            case GLFW_KEY_W: camera_offset.y += pan_speed; break;
            case GLFW_KEY_S: camera_offset.y -= pan_speed; break;
            case GLFW_KEY_A: camera_offset.x -= pan_speed; break;
            case GLFW_KEY_D: camera_offset.x += pan_speed; break;
            case GLFW_KEY_R: camera_offset = Vector2(0,0); zoom = 1.0f; break;
        }
    }
}

Vector2 Renderer::screen_to_world(double screen_x, double screen_y) const {
    // Преобразование с учётом текущей камеры
    float left   = -200.0f / zoom + camera_offset.x;
    float right  =  200.0f / zoom + camera_offset.x;
    float bottom = -200.0f / zoom + camera_offset.y;
    float top    =  200.0f / zoom + camera_offset.y;

    float world_x = left + (screen_x / window_width) * (right - left);
    float world_y = bottom + (1.0 - screen_y / window_height) * (top - bottom);
    return Vector2(world_x, world_y);
}

bool Renderer::should_close() const {
    return glfwWindowShouldClose(window);
}

void Renderer::poll_events() {
    glfwPollEvents();
}
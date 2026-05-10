#define NOMINMAX
#include "renderer.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>
#include <corecrt_math_defines.h>

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

static void add_triangle(std::vector<ConnectionVertex>& v, Vector2 p1, Vector2 p2, Vector2 p3, float r, float g, float b) {
    v.push_back({p1.x, p1.y, r, g, b});
    v.push_back({p2.x, p2.y, r, g, b});
    v.push_back({p3.x, p3.y, r, g, b});
}
static void add_circle(std::vector<ConnectionVertex>& v, Vector2 center, float radius, int seg, float r, float g, float b) {
    for (int i = 0; i < seg; ++i) {
        double a1 = 2.0*M_PI*i/seg, a2 = 2.0*M_PI*(i+1)/seg;
        add_triangle(v, center,
                     center + Vector2{static_cast<float>(radius*cos(a1)), static_cast<float>(radius*sin(a1))},
                     center + Vector2{static_cast<float>(radius*cos(a2)), static_cast<float>(radius*sin(a2))}, r, g, b);
    }
}
static void add_line(std::vector<ConnectionVertex>& v, Vector2 p1, Vector2 p2, float r, float g, float b) {
    v.push_back({p1.x, p1.y, r, g, b});
    v.push_back({p2.x, p2.y, r, g, b});
}

Renderer::Renderer(int width, int height) : window_width(width), window_height(height), window(nullptr) {}

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

bool Renderer::initialize(FlockSimulation& simulation) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(window_width, window_height, "Flocking Simulation - GPU Renderer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) { std::cerr << "Failed to initialize GLEW" << std::endl; return false; }
    glViewport(0, 0, window_width, window_height);

    shader_program = load_shaders(vertex_shader_source, fragment_shader_source);
    if (!shader_program) return false;

    glGenVertexArrays(1, &vao_agents);
    glGenBuffers(1, &vbo_agents);
    glBindVertexArray(vao_agents);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_agents);
    glBufferData(GL_ARRAY_BUFFER, simulation.get_agent_count() * 3 * sizeof(ConnectionVertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &vao_beta);
    glGenBuffers(1, &vbo_beta);
    glBindVertexArray(vao_beta);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_beta);
    glBufferData(GL_ARRAY_BUFFER, simulation.get_max_beta_agents() * 6 * sizeof(ConnectionVertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &vao_connections);
    glGenBuffers(1, &vbo_connections);
    glBindVertexArray(vao_connections);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_connections);
    glBufferData(GL_ARRAY_BUFFER, simulation.get_max_connection_vertices() * sizeof(ConnectionVertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &vao_obstacles);
    glGenBuffers(1, &vbo_obstacles);
    glGenVertexArrays(1, &vao_target);
    glGenBuffers(1, &vbo_target);

    glGenVertexArrays(1, &vao_grid);
    glGenBuffers(1, &vbo_grid);
    build_grid_geometry();

    simulation.register_gl_buffers(vbo_agents, vbo_beta, vbo_connections);

    last_frame_time = std::chrono::steady_clock::now();
    std::cout << "Renderer initialized (GPU-driven, optimized)" << std::endl;
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
        std::cerr << "Vertex shader error: " << info << std::endl;
        return 0;
    }
    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragment_source, nullptr);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(fragment, 512, nullptr, info);
        std::cerr << "Fragment shader error: " << info << std::endl;
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
        std::cerr << "Shader link error: " << info << std::endl;
        return 0;
    }
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return program;
}

void Renderer::get_visible_bounds(float& left, float& right, float& bottom, float& top) const {
    float aspect = (float)window_width / window_height;
    float half_h = WORLD_BOUNDARY / zoom;
    float half_w = half_h * aspect;
    left   = camera_offset.x - half_w;
    right  = camera_offset.x + half_w;
    bottom = camera_offset.y - half_h;
    top    = camera_offset.y + half_h;
}

void Renderer::render(FlockSimulation& simulation) {
    glViewport(0, 0, window_width, window_height);

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
        std::ostringstream title;
        title << "Flocking | FPS: " << (int)fps
              << " | Frame: " << std::fixed << std::setprecision(2) << frame_time_ms << " ms"
              << " | Sim: " << sim_time_ms << " ms";
        glfwSetWindowTitle(window, title.str().c_str());
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float left, right, bottom, top;
    get_visible_bounds(left, right, bottom, top);
    float projection[16] = {
        2.0f/(right-left), 0, 0, 0,
        0, 2.0f/(top-bottom), 0, 0,
        0, 0, -1, 0,
        -(right+left)/(right-left), -(top+bottom)/(top-bottom), 0, 1
    };
    float view[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    glUseProgram(shader_program);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "uProjection"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "uView"), 1, GL_FALSE, view);

    simulation.fill_vbos();

    if (obstacles_dirty) {
        build_obstacles_geometry(simulation.get_obstacles());
        obstacles_dirty = false;
    }
    if (target_dirty) {
        build_target_geometry(simulation.get_target(), simulation.is_target_enabled());
        target_dirty = false;
    }

    // Сетка
    glBindVertexArray(vao_grid);
    glDrawArrays(GL_LINES, 0, grid_vertex_count);

    // Связи
    if (simulation.is_connections_display_enabled() && simulation.get_connection_vertex_count() > 0) {
        glBindVertexArray(vao_connections);
        glDrawArrays(GL_LINES, 0, simulation.get_connection_vertex_count());
    }

    // Препятствия
    glBindVertexArray(vao_obstacles);
    int obs_bytes;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacles);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &obs_bytes);
    glDrawArrays(GL_TRIANGLES, 0, obs_bytes / sizeof(ConnectionVertex));

    // Цель
    if (simulation.is_target_enabled()) {
        glBindVertexArray(vao_target);
        int tgt_bytes;
        glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &tgt_bytes);
        glDrawArrays(GL_LINES, 0, tgt_bytes / sizeof(ConnectionVertex));
    }

    // β-агенты
    if (simulation.is_beta_display_enabled() && simulation.get_beta_count() > 0) {
        glBindVertexArray(vao_beta);
        glDrawArrays(GL_TRIANGLES, 0, simulation.get_beta_count() * 6);
    }

    // Агенты
    glBindVertexArray(vao_agents);
    glDrawArrays(GL_TRIANGLES, 0, simulation.get_agent_count() * 3);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    glfwSwapBuffers(window);
}

void Renderer::build_obstacles_geometry(const std::vector<Obstacle>& obstacles) {
    std::vector<ConnectionVertex> verts;
    for (const auto& obs : obstacles) {
        add_circle(verts, obs.position, obs.radius, 32, 0.9f, 0.2f, 0.2f);
    }
    glBindVertexArray(vao_obstacles);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacles);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(ConnectionVertex), verts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_target_geometry(const Vector2& target, bool enabled) {
    std::vector<ConnectionVertex> verts;
    if (!enabled) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        return;
    }
    add_line(verts, target + Vector2{-8,0}, target + Vector2{8,0}, 0.2f, 0.9f, 0.2f);
    add_line(verts, target + Vector2{0,-8}, target + Vector2{0,8}, 0.2f, 0.9f, 0.2f);
    const int seg = 16;
    for (int i = 0; i < seg; ++i) {
        double a1 = 2.0*M_PI*i/seg, a2 = 2.0*M_PI*(i+1)/seg;
        add_line(verts, target + Vector2{static_cast<float>(12*cos(a1)), static_cast<float>(12*sin(a1))},
                        target + Vector2{static_cast<float>(12*cos(a2)), static_cast<float>(12*sin(a2))}, 0.2f, 0.9f, 0.2f);
    }
    glBindVertexArray(vao_target);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_target);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(ConnectionVertex), verts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::build_grid_geometry() {
    std::vector<ConnectionVertex> verts;
    float step = 20.0f;
    float range = WORLD_BOUNDARY;
    for (float x = -range; x <= range; x += step)
        add_line(verts, {x, -range}, {x, range}, 0.3f, 0.3f, 0.3f);
    for (float y = -range; y <= range; y += step)
        add_line(verts, {-range, y}, {range, y}, 0.3f, 0.3f, 0.3f);
    grid_vertex_count = verts.size();
    glBindVertexArray(vao_grid);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(ConnectionVertex), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::setup_callbacks(FlockSimulation* sim) {
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int b, int a, int m) { ((Renderer*)glfwGetWindowUserPointer(w))->on_mouse_button(b,a,m); });
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) { ((Renderer*)glfwGetWindowUserPointer(w))->on_cursor_pos(x,y); });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double x, double y) { ((Renderer*)glfwGetWindowUserPointer(w))->on_scroll(x,y); });
    glfwSetKeyCallback(window, [](GLFWwindow* w, int k, int s, int a, int m) { ((Renderer*)glfwGetWindowUserPointer(w))->on_key(k,a,m); });
}
void Renderer::on_mouse_button(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) { panning = true; glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y); }
        else if (action == GLFW_RELEASE) panning = false;
    }
}
void Renderer::on_cursor_pos(double x, double y) {
    if (panning) {
        float left, right, bottom, top;
        get_visible_bounds(left, right, bottom, top);
        float vis_w = right - left, vis_h = top - bottom;
        float dx = (float)(x - last_mouse_x) / window_width * vis_w;
        float dy = (float)(y - last_mouse_y) / window_height * vis_h;
        camera_offset.x -= dx;
        camera_offset.y += dy;
        last_mouse_x = x; last_mouse_y = y;
    }
}
void Renderer::on_scroll(double, double yoffset) {
    zoom *= (1.0f + (float)yoffset * 0.1f);
    zoom = std::max(0.2f, std::min(zoom, (float)WORLD_BOUNDARY / 100.0f));
}
void Renderer::on_key(int key, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        float speed = 10.0f / zoom;
        switch (key) {
            case GLFW_KEY_W: camera_offset.y += speed; break;
            case GLFW_KEY_S: camera_offset.y -= speed; break;
            case GLFW_KEY_A: camera_offset.x -= speed; break;
            case GLFW_KEY_D: camera_offset.x += speed; break;
            case GLFW_KEY_R: camera_offset = {0,0}; zoom = 1.0f; break;
        }
    }
}
Vector2 Renderer::screen_to_world(double sx, double sy) const {
    float left, right, bottom, top;
    get_visible_bounds(left, right, bottom, top);
    float wx = left + (float)(sx / window_width) * (right - left);
    float wy = bottom + (float)(1.0 - sy / window_height) * (top - bottom);
    return {wx, wy};
}
bool Renderer::should_close() const { return glfwWindowShouldClose(window); }
void Renderer::poll_events() { glfwPollEvents(); }
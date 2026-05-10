#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

int get_console_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col;
#endif
}

#include "renderer.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

static bool adding_obstacles = false;
static bool setting_target = true;

void print_simulation_info(const FlockSimulation& simulation) {
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 60 == 0) {
        std::ostringstream oss;
        oss << "=== SIMULATION INFO ===";
        oss << " | Agents: " << simulation.get_agent_count();
        oss << " | Obstacles: " << simulation.get_obstacles().size();
        oss << " | Beta-agents: " << simulation.get_beta_count();
        oss << " | Target: " << (simulation.is_target_enabled() ? "ON" : "OFF");
        oss << " | Beta-display: " << (simulation.is_beta_display_enabled() ? "ON" : "OFF");
        oss << " | Connections: " << (simulation.is_connections_display_enabled() ? "ON" : "OFF");
        oss << " | Mode: " << (setting_target ? "SET TARGET" : "ADD OBSTACLES");
        std::string info = oss.str();
        int w = get_console_width();
        if (info.length() > w) info = info.substr(0, w-3) + "...";
        else info += std::string(w - info.length(), ' ');
        std::cout << "\r" << info << std::flush;
    }
}

struct AppContext { FlockSimulation* sim; Renderer* renderer; };

int main() {
    std::cout << "Starting Flocking Simulation (GPU-driven, optimized)..." << std::endl;
    FlockSimulation simulation;
    Renderer renderer(1000, 800);

    if (!renderer.initialize(simulation)) {
        std::cerr << "Failed to initialize renderer!" << std::endl;
        return -1;
    }
    simulation.start();

    AppContext ctx{&simulation, &renderer};
    glfwSetWindowUserPointer(renderer.get_window(), &ctx);
    simulation.set_target({0,0});

    glfwSetFramebufferSizeCallback(renderer.get_window(), [](GLFWwindow* w, int width, int height) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(w));
        if (ctx && ctx->renderer) ctx->renderer->update_window_size(width, height);
    });
    glfwSetMouseButtonCallback(renderer.get_window(), [](GLFWwindow* w, int button, int action, int mods) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(w));
        if (!ctx) return;
        ctx->renderer->on_mouse_button(button, action, mods);
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            double x, y;
            glfwGetCursorPos(w, &x, &y);
            Vector2 wp = ctx->renderer->screen_to_world(x, y);
            if (setting_target) {
                ctx->sim->set_target(wp);
                ctx->renderer->mark_target_dirty();
            } else if (adding_obstacles) {
                ctx->sim->add_obstacle(wp, 10.0 + (rand()%10));
                ctx->renderer->mark_obstacles_dirty();
            }
        }
    });
    glfwSetKeyCallback(renderer.get_window(), [](GLFWwindow* w, int key, int scancode, int action, int mods) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(w));
        if (!ctx) return;
        ctx->renderer->on_key(key, action, mods);
        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_T:
                    setting_target = true; adding_obstacles = false;
                    ctx->sim->enable_target();
                    ctx->renderer->mark_target_dirty();
                    break;
                case GLFW_KEY_O:
                    setting_target = false; adding_obstacles = true;
                    break;
                case GLFW_KEY_C:
                    ctx->sim->clear_obstacles();
                    ctx->renderer->mark_obstacles_dirty();
                    break;
                case GLFW_KEY_B: ctx->sim->toggle_beta_display(); break;
                case GLFW_KEY_X:
                    ctx->sim->remove_target();
                    ctx->renderer->mark_target_dirty();
                    break;
                case GLFW_KEY_G: ctx->sim->toggle_connections(); break;
                case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
                case GLFW_KEY_H:
                    std::cout << "\n=== CONTROLS ===" << std::endl;
                    std::cout << "T - set target | O - add obstacles | C - clear obstacles" << std::endl;
                    std::cout << "B - toggle beta | X - remove target | G - toggle connections" << std::endl;
                    std::cout << "ESC - exit" << std::endl;
                    break;
            }
        }
    });
    glfwSetCursorPosCallback(renderer.get_window(), [](GLFWwindow* w, double x, double y) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(w));
        if (ctx && ctx->renderer) ctx->renderer->on_cursor_pos(x, y);
    });
    glfwSetScrollCallback(renderer.get_window(), [](GLFWwindow* w, double xoff, double yoff) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(w));
        if (ctx && ctx->renderer) ctx->renderer->on_scroll(xoff, yoff);
    });

    auto last_sim_time = std::chrono::steady_clock::now();
    while (!renderer.should_close()) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_sim_time).count();
        dt = std::min(dt, 0.1f);
        auto sim_start = std::chrono::steady_clock::now();
        if (simulation.is_running()) simulation.step(dt);
        auto sim_end = std::chrono::steady_clock::now();
        renderer.set_sim_time(std::chrono::duration<double, std::milli>(sim_end - sim_start).count());
        last_sim_time = now;

        renderer.render(simulation);
        renderer.poll_events();
    }
    simulation.stop();
    std::cout << "\nSimulation stopped." << std::endl;
    return 0;
}
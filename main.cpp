#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// Функция для получения ширины консоли
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

// Глобальные переменные для управления режимами
static bool adding_obstacles = false;
static bool setting_target = true;

// Функция для вывода информации о состоянии симуляции
void print_simulation_info(const FlockSimulation& simulation) {
    static int frame_count = 0;
    frame_count++;
    
    if (frame_count % 60 == 0) {
        std::ostringstream oss;
        oss << "=== SIMULATION INFO ===";
        oss << " | Agents: " << simulation.get_agents().size();
        oss << " | Obstacles: " << simulation.get_obstacles().size();
        oss << " | Beta-agents: " << simulation.get_beta_agents().size();
        oss << " | Target: " << (simulation.is_target_enabled() ? "ON" : "OFF");
        oss << " | Beta-display: " << (simulation.is_beta_display_enabled() ? "ON" : "OFF");
        oss << " | Connections: " << (simulation.is_connections_display_enabled() ? "ON" : "OFF");
        oss << " | Mode: " << (setting_target ? "SET TARGET" : "ADD OBSTACLES");
        
        std::string info_str = oss.str();
        
        int console_width = get_console_width();
        if (info_str.length() > console_width) {
            info_str = info_str.substr(0, console_width - 3) + "...";
        } else {
            info_str += std::string(console_width - info_str.length(), ' ');
        }
        
        std::cout << "\r" << info_str << std::flush;
    }
}

struct AppContext {
    FlockSimulation* sim;
    Renderer* renderer;
};

int main() {
    std::cout << "Starting Flocking Simulation (Algorithm 3)..." << std::endl;
    
    // Инициализация рендерера
    Renderer renderer(1000, 800);
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer!" << std::endl;
        return -1;
    }
    
    // Создание симуляции
    FlockSimulation simulation;
    simulation.start();

    AppContext ctx;
    ctx.sim = &simulation;
    ctx.renderer = &renderer;
    glfwSetWindowUserPointer(renderer.get_window(), &ctx);
    
    // Устанавливаем начальную цель в центре
    simulation.set_target(Vector2(0, 0));
    
    // Колбэк для изменения размеров окна (обновляет вьюпорт и размеры рендерера)
    glfwSetFramebufferSizeCallback(renderer.get_window(), [](GLFWwindow* window, int width, int height) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(window));
        if (ctx && ctx->renderer) {
            ctx->renderer->update_window_size(width, height);
        }
    });
    
    // Колбэк для мыши
    glfwSetMouseButtonCallback(renderer.get_window(), [](GLFWwindow* window, int button, int action, int mods) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(window));
        if (!ctx) return;
        // Обработка камеры (правая кнопка)
        if (ctx->renderer) ctx->renderer->on_mouse_button(button, action, mods);
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            FlockSimulation* sim = ctx->sim;
            Renderer* rend = ctx->renderer;
            if (sim && rend) {
                double x, y;
                glfwGetCursorPos(window, &x, &y);
                Vector2 world_pos = rend->screen_to_world(x, y);
                if (setting_target) {
                    sim->set_target(world_pos);
                } else if (adding_obstacles) {
                    double radius = 10.0 + (rand() % 10);
                    sim->add_obstacle(world_pos, radius);
                }
            }
        }
    });
    
    // Колбэк для клавиш
    glfwSetKeyCallback(renderer.get_window(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(window));
        if (!ctx) return;
        // Обработка камеры
        if (ctx->renderer) ctx->renderer->on_key(key, action, mods);
        if (action == GLFW_PRESS) {
            FlockSimulation* sim = ctx->sim;
            Renderer* rend = ctx->renderer;
            
            switch (key) {
                case GLFW_KEY_T:
                    setting_target = true;
                    adding_obstacles = false;
                    sim->enable_target();
                    std::cout << "\nMODE: Set Target (click to set flock target)" << std::endl;
                    break;
                    
                case GLFW_KEY_O:
                    setting_target = false;
                    adding_obstacles = true;
                    std::cout << "\nMODE: Add Obstacles (click to place obstacles)" << std::endl;
                    break;
                    
                case GLFW_KEY_C:
                    if (sim) {
                        sim->clear_obstacles();
                        std::cout << "\nAll obstacles cleared" << std::endl;
                    }
                    break;
                    
                case GLFW_KEY_B:
                    if (sim) {
                        sim->toggle_beta_display();
                        std::cout << "\nbeta-AGENTS: " << (sim->is_beta_display_enabled() ? "VISIBLE" : "HIDDEN") << std::endl;
                    }
                    break;
                    
                case GLFW_KEY_X:
                    if (sim) {
                        sim->remove_target();
                        std::cout << "\nTARGET REMOVED - Flocking without navigation" << std::endl;
                        std::cout << "Agents will maintain swarm behavior and obstacle avoidance only" << std::endl;
                    }
                    break;
                    
                case GLFW_KEY_G:
                    if (sim) {
                        sim->toggle_connections();
                        std::cout << "\nCONNECTIONS: " << (sim->is_connections_display_enabled() ? "SHOW" : "HIDE") << std::endl;
                    }
                    break;
                    
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    break;
                    
                case GLFW_KEY_H:
                    std::cout << "\n=== FLOCKING SIMULATION CONTROLS ===" << std::endl;
                    std::cout << "T - Set target mode (click to set flock target)" << std::endl;
                    std::cout << "O - Add obstacle mode (click to place obstacles)" << std::endl;
                    std::cout << "C - Clear all obstacles" << std::endl;
                    std::cout << "B - Toggle beta-agents display" << std::endl;
                    std::cout << "X - Remove target (swarm only mode)" << std::endl;
                    std::cout << "G - Toggle connections display" << std::endl;
                    std::cout << "H - Show this help" << std::endl;
                    std::cout << "ESC - Exit" << std::endl;
                    std::cout << "=====================================" << std::endl;
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
    
    // Главный цикл
    auto last_sim_time = std::chrono::steady_clock::now();
    
    std::cout << "\n=== FLOCKING SIMULATION CONTROLS ===" << std::endl;
    std::cout << "T - Set target mode (click to set flock target)" << std::endl;
    std::cout << "O - Add obstacle mode (click to place obstacles)" << std::endl;
    std::cout << "C - Clear all obstacles" << std::endl;
    std::cout << "B - Toggle beta-agents display" << std::endl;
    std::cout << "X - Remove target (swarm only mode)" << std::endl;
    std::cout << "G - Toggle connections display" << std::endl;
    std::cout << "H - Show this help" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Current mode: " << (setting_target ? "SET TARGET" : "ADD OBSTACLES") << std::endl;
    std::cout << "Target: " << (simulation.is_target_enabled() ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "beta-agents display: " << (simulation.is_beta_display_enabled() ? "ON" : "OFF") << std::endl;
    std::cout << "Connections display: " << (simulation.is_connections_display_enabled() ? "ON" : "OFF") << std::endl;
    
    while (!renderer.should_close()) {
        auto current_time = std::chrono::steady_clock::now();
        
        // Симуляция с фиксированным шагом (макс 0.1 сек)
        auto sim_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - last_sim_time);
        double delta_time = std::min(sim_elapsed.count(), 0.1);
        
        auto sim_start = std::chrono::steady_clock::now();

        if (simulation.is_running()) {
            simulation.step(delta_time);
        }
        auto sim_end = std::chrono::steady_clock::now();
        double sim_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();
        renderer.set_sim_time(sim_ms);

        last_sim_time = current_time;

        renderer.render(simulation);
        renderer.poll_events();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    simulation.stop();
    std::cout << "\nSimulation stopped. Goodbye!" << std::endl;
    
    return 0;
}
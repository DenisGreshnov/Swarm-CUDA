#pragma once

#ifdef _WIN32
#include <windows.h>   // <-- обязательно перед GL/gl.h на Windows
#endif

#include <vector>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WORLD_BOUNDARY 4000.0

// 2D вектор (CPU + GPU)
struct Vector2 {
    double x, y;
    __host__ __device__ Vector2(double x = 0, double y = 0) : x(x), y(y) {}
    __host__ __device__ Vector2 operator+(const Vector2& o) const { return Vector2(x+o.x, y+o.y); }
    __host__ __device__ Vector2 operator-(const Vector2& o) const { return Vector2(x-o.x, y-o.y); }
    __host__ __device__ Vector2 operator*(double s) const { return Vector2(x*s, y*s); }
    __host__ __device__ double dot(const Vector2& o) const { return x*o.x + y*o.y; }
    __host__ __device__ double length() const { return sqrt(x*x + y*y); }
    __host__ __device__ Vector2 normalized() const {
        double l = length();
        return (l < 1e-10) ? Vector2(0,0) : Vector2(x/l, y/l);
    }
};

// Структуры данных на GPU
struct Agent {
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;
};

struct Obstacle {
    Vector2 position;
    double radius;
    bool is_wall;
    Vector2 wall_normal;
};

struct BetaAgent {
    Vector2 position;
    Vector2 velocity;
};

// Вершина для рендеринга (используется и в рендерере, и в CUDA-ядрах)
struct ConnectionVertex {
    float x, y;
    float r, g, b;
};
using Vertex = ConnectionVertex;   // единообразное имя

// Параметры симуляции (константы для ядер)
struct SimParams {
    double desired_distance;
    double interaction_range;
    double obstacle_range;
    double c1_alpha, c2_alpha;
    double c1_beta,  c2_beta;
    double c1_gamma, c2_gamma;
    double epsilon;
    double h_alpha, h_beta;
    double a, b;
    bool use_gamma_target;
    Vector2 gamma_target;
    Vector2 gamma_velocity;
};

class FlockSimulation {
public:
    FlockSimulation();
    ~FlockSimulation();

    // Основной шаг (только физика, без рендеринга)
    void step(double delta_time);

    // Заполнение VBO прямо на GPU (вызывается перед отрисовкой)
    void fill_vbos();

    // Регистрация OpenGL-буферов для CUDA
    void register_gl_buffers(GLuint vbo_agents, GLuint vbo_beta, GLuint vbo_connections);
    void unregister_gl_buffers();

    // Управление препятствиями и целью
    void add_obstacle(const Vector2& pos, double radius);
    void clear_obstacles();
    void set_target(const Vector2& target);
    void remove_target()   { params.use_gamma_target = false; sync_params_to_gpu(); }
    void enable_target()   { params.use_gamma_target = true;  sync_params_to_gpu(); }
    std::vector<Obstacle> get_obstacles() const { return h_obstacles; }

    // Характеристики для рендеринга (без копирования тяжёлых данных)
    int  get_agent_count()          const { return num_agents; }
    int  get_beta_count()           const { return h_beta_count; }
    int  get_max_beta_agents()      const { return max_beta_agents; }
    int  get_max_connection_vertices() const { return max_connection_vertices; }
    int  get_connection_vertex_count() const { return h_connection_count; }
    Vector2 get_target()            const { return params.gamma_target; }

    // Флаги отображения
    void toggle_beta_display()  { show_beta_agents = !show_beta_agents; }
    void toggle_connections()   { show_connections = !show_connections; }
    bool is_target_enabled() const          { return params.use_gamma_target; }
    bool is_beta_display_enabled() const    { return show_beta_agents; }
    bool is_connections_display_enabled() const { return show_connections; }

    // Управление запуском
    void start() { running = true; }
    void stop()  { running = false; }
    bool is_running() const { return running; }

private:
    SimParams params;
    SimParams* d_params = nullptr;

    // Данные на GPU
    Agent*     d_agents = nullptr;
    Obstacle*  d_obstacles = nullptr;
    BetaAgent* d_beta_agents = nullptr;
    int*       d_beta_count = nullptr;

    // Пространственное хэширование
    int* d_hashes = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;
    int grid_resolution;
    double cell_size;
    int total_cells;

    // Ресурсы CUDA-OpenGL interop
    cudaGraphicsResource_t cuda_vbo_agents = nullptr;
    cudaGraphicsResource_t cuda_vbo_beta = nullptr;
    cudaGraphicsResource_t cuda_vbo_connections = nullptr;

    // Счетчик вершин связей (единственная копия на CPU)
    int h_connection_count = 0;
    int* d_connection_count = nullptr;

    int max_connection_vertices;

    // CPU‑копии β‑агентов (нужны для физики, но не для рендеринга)
    std::vector<BetaAgent> h_beta_agents;
    int h_beta_count = 0;

    //CPU‑хранилище препятствий (для добавления/очистки и отрисовки)
    std::vector<Obstacle> h_obstacles;

    // Флаги
    bool show_beta_agents = false;
    bool show_connections = false;
    std::atomic<bool> running{false};

    // Размеры массивов
    int num_agents = 100000;
    int num_obstacles = 0;
    int max_obstacles = 1000;
    int max_beta_agents = 2000;

    // Временные CPU‑векторы для инициализации
    std::vector<Agent> h_agents_init;

    // Внутренние методы
    void allocate_gpu_memory();
    void free_gpu_memory();
    void sync_params_to_gpu();
    void copy_agents_to_gpu();       // только при старте
    void copy_obstacles_to_gpu();

    // CUDA-шаги
    void generate_beta_agents();
    void compute_forces();
    void integrate(double delta_time);
    void prepare_spatial_hashing();
};
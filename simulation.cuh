#pragma once

#include <vector>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>

#define WORLD_BOUNDARY 2000.0

// Простой 2D вектор (используется и на CPU, и на GPU)
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

// Структуры данных для GPU
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

// Вершина линии для отрисовки связей (совместима с Vertex в рендерере)
struct ConnectionVertex {
    float x, y;
    float r, g, b;
};

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

// Основной класс симуляции (управляет памятью GPU)
class FlockSimulation {
public:
    FlockSimulation();
    ~FlockSimulation();

    void step(double delta_time);

    // Управление препятствиями и целью
    void add_obstacle(const Vector2& pos, double radius);
    void clear_obstacles();
    void set_target(const Vector2& target);
    void remove_target()   { params.use_gamma_target = false; sync_params_to_gpu(); }
    void enable_target()   { params.use_gamma_target = true;  sync_params_to_gpu(); }

    // Получение копий для рендеринга
    std::vector<Agent>      get_agents()      const;
    std::vector<Obstacle>   get_obstacles()   const;
    std::vector<BetaAgent>  get_beta_agents() const;
    Vector2                 get_target()      const { return params.gamma_target; }

    // Получение геометрии связей (генерируется на GPU)
    const ConnectionVertex* get_connection_vertices() const { return h_connection_vertices.data(); }
    int get_connection_vertex_count() const { return h_connection_count; }

    // Управление отображением
    void toggle_beta_display()  { show_beta_agents = !show_beta_agents; }
    void toggle_connections()   { show_connections = !show_connections; }
    bool is_target_enabled() const          { return params.use_gamma_target; }
    bool is_beta_display_enabled() const    { return show_beta_agents; }
    bool is_connections_display_enabled() const { return show_connections; }

    // Геттеры параметров для рендеринга связей
    double get_interaction_range() const { return params.interaction_range; }
    double get_obstacle_range()   const { return params.obstacle_range; }

    void start() { running = true; }
    void stop()  { running = false; }
    bool is_running() const { return running; }

private:
    // Параметры на CPU и GPU
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

    // Связи для рендеринга (GPU)
    ConnectionVertex* d_connection_vertices = nullptr;
    int* d_connection_count = nullptr;
    int max_connection_vertices;

    // Копии на CPU (для рендеринга)
    std::vector<Agent>      h_agents;
    std::vector<Obstacle>   h_obstacles;
    std::vector<BetaAgent>  h_beta_agents;
    int h_beta_count = 0;

    std::vector<ConnectionVertex> h_connection_vertices;
    int h_connection_count = 0;

    // Флаги
    bool show_beta_agents = false;
    bool show_connections = false;
    std::atomic<bool> running{false};

    // Размеры массивов на GPU
    int num_agents = 75000;
    int num_obstacles = 0;
    int max_obstacles = 1000;
    int max_beta_agents = 2000;

    // Вспомогательные методы
    void allocate_gpu_memory();
    void free_gpu_memory();
    void sync_params_to_gpu();
    void copy_agents_to_gpu();
    void copy_obstacles_to_gpu();
    void copy_beta_agents_from_gpu(int count);

    // Внутренние функции для CUDA-шага
    void generate_beta_agents();
    void compute_forces();
    void integrate(double delta_time);
    void prepare_spatial_hashing();
    void build_connections();   // генерация геометрии связей на GPU
};
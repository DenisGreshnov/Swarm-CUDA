#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include <vector>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WORLD_BOUNDARY 10000.0f

// 2D вектор (агрегат, конструктор удалён)
struct Vector2 {
    float x, y;
    // методы, возвращающие новый вектор, используют агрегатную инициализацию {}
    __host__ __device__ Vector2 operator+(const Vector2& o) const { return {x+o.x, y+o.y}; }
    __host__ __device__ Vector2 operator-(const Vector2& o) const { return {x-o.x, y-o.y}; }
    __host__ __device__ Vector2 operator*(float s) const { return {x*s, y*s}; }
    __host__ __device__ float dot(const Vector2& o) const { return x*o.x + y*o.y; }
    __host__ __device__ float length() const { return sqrtf(x*x + y*y); }
    __host__ __device__ Vector2 normalized() const {
        float l = length();
        return (l < 1e-10f) ? Vector2{0,0} : Vector2{x/l, y/l};
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
    float radius;
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
using Vertex = ConnectionVertex;

// Параметры симуляции с предвычисленными константами
struct SimParams {
    // Исходные параметры
    float desired_distance;
    float interaction_range;
    float obstacle_range;
    float c1_alpha, c2_alpha;
    float c1_beta,  c2_beta;
    float c1_gamma, c2_gamma;
    float epsilon;
    float h_alpha, h_beta;
    float a, b;
    bool use_gamma_target;
    Vector2 gamma_target;
    Vector2 gamma_velocity;

    // Предвычисленные величины
    float sigma_d_alpha;
    float sigma_r_alpha;
    float sigma_d_beta;
    float c_phi;
};

// Декларация постоянной памяти (c_params – агрегат, инициализируется нулями)
//extern __constant__ SimParams c_params;

class FlockSimulation {
public:
    FlockSimulation();
    ~FlockSimulation();

    void step(float delta_time);
    void fill_vbos();

    void register_gl_buffers(GLuint vbo_agents, GLuint vbo_beta, GLuint vbo_connections);
    void unregister_gl_buffers();

    void add_obstacle(const Vector2& pos, float radius);
    void clear_obstacles();
    void set_target(const Vector2& target);
    void remove_target()   { params.use_gamma_target = false; sync_params_to_gpu(); }
    void enable_target()   { params.use_gamma_target = true;  sync_params_to_gpu(); }
    std::vector<Obstacle> get_obstacles() const { return h_obstacles; }

    int  get_agent_count()          const { return num_agents; }
    int  get_beta_count()           const { return h_beta_count; }
    int  get_max_beta_agents()      const { return max_beta_agents; }
    int  get_max_connection_vertices() const { return max_connection_vertices; }
    int  get_connection_vertex_count() const { return h_connection_count; }
    Vector2 get_target()            const { return params.gamma_target; }

    void toggle_beta_display()  { show_beta_agents = !show_beta_agents; }
    void toggle_connections()   { show_connections = !show_connections; }
    bool is_target_enabled() const          { return params.use_gamma_target; }
    bool is_beta_display_enabled() const    { return show_beta_agents; }
    bool is_connections_display_enabled() const { return show_connections; }

    void start() { running = true; }
    void stop()  { running = false; }
    bool is_running() const { return running; }

private:
    SimParams params;

    // Данные на GPU (указатели)
    Agent*     d_agents = nullptr;
    Obstacle*  d_obstacles = nullptr;
    BetaAgent* d_beta_agents = nullptr;
    int*       d_beta_count = nullptr;

    // Пространственное хэширование
    int* d_hashes = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;
    int grid_resolution;
    float cell_size;
    int total_cells;

    // Для оптимизированного построения связей
    int* d_block_conn_counts = nullptr;
    int* d_block_conn_offsets = nullptr;
    int max_blocks_agents;

    // CUDA-OpenGL interop
    cudaGraphicsResource_t cuda_vbo_agents = nullptr;
    cudaGraphicsResource_t cuda_vbo_beta = nullptr;
    cudaGraphicsResource_t cuda_vbo_connections = nullptr;

    // Счетчик вершин связей
    int h_connection_count = 0;
    int* d_connection_count = nullptr;
    int max_connection_vertices;

    // CPU‑копия количества β‑агентов
    int h_beta_count = 0;

    // CPU‑хранилище препятствий
    std::vector<Obstacle> h_obstacles;

    // Флаги
    bool show_beta_agents = false;
    bool show_connections = false;
    std::atomic<bool> running{false};

    // Размеры массивов
    int num_agents = 2000000;
    int num_obstacles = 0;
    int max_obstacles = 1000;
    int max_beta_agents = 2000;

    // Временные CPU‑векторы для инициализации
    std::vector<Agent> h_agents_init;

    // Внутренние методы
    void allocate_gpu_memory();
    void free_gpu_memory();
    void sync_params_to_gpu();
    void copy_agents_to_gpu();
    void copy_obstacles_to_gpu();

    // CUDA-шаги
    void generate_beta_agents();
    void compute_forces();
    void integrate(float delta_time);
    void prepare_spatial_hashing();
};
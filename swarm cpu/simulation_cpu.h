#pragma once
#include <vector>
#include <atomic>
#include <GL/glew.h>
#include "common_types.h"

struct SimParams {
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

    // предвычисленные величины
    float sigma_d_alpha;
    float sigma_r_alpha;
    float sigma_d_beta;
    float c_phi;
};

class FlockSimulationCPU {
public:
    FlockSimulationCPU();
    ~FlockSimulationCPU() = default;

    void step(float delta_time);
    void fill_vbos();

    void register_gl_buffers(GLuint vbo_agents, GLuint vbo_beta, GLuint vbo_connections);

    void add_obstacle(const Vector2& pos, float radius);
    void clear_obstacles();
    void set_target(const Vector2& target);
    void remove_target();
    void enable_target();

    std::vector<Obstacle> get_obstacles() const { return h_obstacles; }

    int get_agent_count()          const { return num_agents; }
    int get_beta_count()           const { return static_cast<int>(beta_agents.size()); }
    int get_max_beta_agents()      const { return max_beta_agents; }
    int get_max_connection_vertices() const { return max_connection_vertices; }
    int get_connection_vertex_count() const { return h_connection_count; }
    Vector2 get_target()           const { return params.gamma_target; }

    void toggle_beta_display()     { show_beta_agents = !show_beta_agents; }
    void toggle_connections()      { show_connections = !show_connections; }
    bool is_target_enabled() const { return params.use_gamma_target; }
    bool is_beta_display_enabled() const    { return show_beta_agents; }
    bool is_connections_display_enabled() const { return show_connections; }

    void start() { running = true; }
    void stop()  { running = false; }
    bool is_running() const { return running; }

private:
    // Параметры
    SimParams params;

    // Агенты (CPU)
    std::vector<Agent> agents;
    // Препятствия (CPU)
    std::vector<Obstacle> h_obstacles;
    // β‑агенты (CPU)
    std::vector<BetaAgent> beta_agents;
    int max_beta_agents = 2000;

    // Пространственное хэширование
    int grid_resolution;
    float cell_size;
    int total_cells;
    std::vector<int> cell_start;
    std::vector<int> cell_end;
    std::vector<int> agent_indices;   // индексы агентов, отсортированные по ячейке

    // Связи (для визуализации)
    int h_connection_count = 0;
    int max_connection_vertices;

    // OpenGL буферы
    GLuint vbo_agents = 0;
    GLuint vbo_beta = 0;
    GLuint vbo_connections = 0;

    // Флаги
    std::atomic<bool> running{false};
    bool show_beta_agents = false;
    bool show_connections = false;

    // Константы
    static constexpr int num_agents = 1000000;
    static constexpr int max_obstacles = 1000;

    void init_agents();
    void sync_params();
    void generate_beta_agents();
    void prepare_spatial_hashing();
    void compute_forces();
    void integrate(float delta_time);

    // Вспомогательные функции для вычисления сил (аналоги GPU‑ядрам)
    float sigma_norm(const Vector2& z, float epsilon) const;
    Vector2 sigma_epsilon(const Vector2& z, float epsilon) const;
    float bump_function(float z, float h) const;
    float sigma1(float s) const;
    float phi_alpha(float z) const;
    float phi_beta(float z) const;
    float alpha_adjacency(const Vector2& qi, const Vector2& qj) const;
    float beta_adjacency(const Vector2& qi, const Vector2& qb) const;

    // Построение геометрии для VBO
    void build_agents_vbo(std::vector<Vertex>& verts);
    void build_beta_vbo(std::vector<Vertex>& verts);
    void build_connections_vbo(std::vector<Vertex>& verts);
};
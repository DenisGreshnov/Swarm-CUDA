#include "simulation_cpu.h"
#include <algorithm>
#include <random>
#include <cstring>
#include <iostream>
#include <cmath>
#include <numeric>

FlockSimulationCPU::FlockSimulationCPU() {
    params.desired_distance   = 10.0f;
    params.interaction_range  = 1.2f * params.desired_distance;
    params.obstacle_range     = 1.2f * 0.6f * params.desired_distance;
    params.c1_alpha = 100.0f;  params.c2_alpha = 10.0f;
    params.c1_beta  = 1000.0f; params.c2_beta  = 10.0f;
    params.c1_gamma = 5.0f;    params.c2_gamma = 0.2f;
    params.epsilon = 0.1f;
    params.h_alpha = 0.2f;     params.h_beta = 0.9f;
    params.a = 1.0f;           params.b = 10.0f;
    params.use_gamma_target = true;
    params.gamma_target = {0.0f, 0.0f};
    params.gamma_velocity = {0.0f, 0.0f};

    float d = params.desired_distance;
    params.sigma_d_alpha = (1.0f / params.epsilon) * (sqrtf(1.0f + params.epsilon * d * d) - 1.0f);
    float r = params.interaction_range;
    params.sigma_r_alpha = (1.0f / params.epsilon) * (sqrtf(1.0f + params.epsilon * r * r) - 1.0f);
    float d_beta = params.desired_distance * 0.6f;
    params.sigma_d_beta  = (1.0f / params.epsilon) * (sqrtf(1.0f + params.epsilon * d_beta * d_beta) - 1.0f);
    params.c_phi = fabsf(params.a - params.b) / sqrtf(4.0f * params.a * params.b);

    init_agents();

    cell_size = params.interaction_range * 1.05f;
    grid_resolution = (int)ceilf(2.0f * WORLD_BOUNDARY / cell_size) + 2;
    total_cells = grid_resolution * grid_resolution;
    cell_start.resize(total_cells);
    cell_end.resize(total_cells);
    agent_indices.resize(num_agents);

    max_connection_vertices = num_agents * 10;
}

void FlockSimulationCPU::init_agents() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis1(-sqrt(num_agents)*params.desired_distance*0.6f, sqrt(num_agents)*params.desired_distance*0.6f);
    std::uniform_real_distribution<float> dis2(-10, 10);
    agents.resize(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        agents[i].position = {dis1(gen), dis1(gen)};
        agents[i].velocity = {dis2(gen), dis2(gen)};
        agents[i].acceleration = {0,0};
    }
}

void FlockSimulationCPU::sync_params() {
    // предвычисленные константы уже в params
}

void FlockSimulationCPU::add_obstacle(const Vector2& pos, float radius) {
    if (h_obstacles.size() >= max_obstacles) return;
    h_obstacles.push_back({pos, radius, false, {0,0}});
}

void FlockSimulationCPU::clear_obstacles() {
    h_obstacles.clear();
}

void FlockSimulationCPU::set_target(const Vector2& target) {
    params.gamma_target = target;
    params.use_gamma_target = true;
}

void FlockSimulationCPU::remove_target() {
    params.use_gamma_target = false;
}

void FlockSimulationCPU::enable_target() {
    params.use_gamma_target = true;
}

// ----- Вспомогательные функции для сил (прямые кальки с GPU) -----
float FlockSimulationCPU::sigma_norm(const Vector2& z, float epsilon) const {
    float n = z.length();
    return (1.0f / epsilon) * (sqrtf(1.0f + epsilon * n * n) - 1.0f);
}

Vector2 FlockSimulationCPU::sigma_epsilon(const Vector2& z, float epsilon) const {
    float n = z.length();
    if (n < 1e-10f) return {0,0};
    return z * (1.0f / sqrtf(1.0f + epsilon * n * n));
}

float FlockSimulationCPU::bump_function(float z, float h) const {
    if (z < h) return 1.0f;
    if (z < 1.0f) return 0.5f * (1.0f + cosf(PI * (z - h) / (1.0f - h)));
    return 0.0f;
}

float FlockSimulationCPU::sigma1(float s) const {
    return s / sqrtf(1.0f + s * s);
}

float FlockSimulationCPU::phi_alpha(float z) const {
    float bump = bump_function(z / params.sigma_r_alpha, params.h_alpha);
    float s = z - params.sigma_d_alpha;
    float phi_s = 0.5f * ((params.a + params.b) * sigma1(s + params.c_phi) + (params.a - params.b));
    return bump * phi_s;
}

float FlockSimulationCPU::phi_beta(float z) const {
    float bump = bump_function(z / params.sigma_d_beta, params.h_beta);
    float s = z - params.sigma_d_beta;
    float action = sigma1(s) - 1.0f;
    return bump * action;
}

float FlockSimulationCPU::alpha_adjacency(const Vector2& qi, const Vector2& qj) const {
    float dist = sigma_norm(qj - qi, params.epsilon);
    return bump_function(dist / params.sigma_r_alpha, params.h_alpha);
}

float FlockSimulationCPU::beta_adjacency(const Vector2& qi, const Vector2& qb) const {
    float dist = sigma_norm(qb - qi, params.epsilon);
    return bump_function(dist / params.sigma_d_beta, params.h_beta);
}

// ----- Основные шаги симуляции -----
void FlockSimulationCPU::generate_beta_agents() {
    beta_agents.clear();
    if (h_obstacles.empty()) return;

    for (int i = 0; i < num_agents; ++i) {
        const Agent& agent = agents[i];
        for (const auto& obs : h_obstacles) {
            Vector2 to_obs = obs.position - agent.position;
            float dist = to_obs.length();
            if (dist < params.obstacle_range + obs.radius) {
                BetaAgent beta;
                if (obs.is_wall) {
                    if (fabsf(obs.position.x - agent.position.x) < fabsf(obs.position.y - agent.position.y)) {
                        beta.position = {agent.position.x, obs.position.y};
                        beta.velocity = {agent.velocity.x, 0};
                    } else {
                        beta.position = {obs.position.x, agent.position.y};
                        beta.velocity = {0, agent.velocity.y};
                    }
                } else {
                    if (dist > 1e-6f) {
                        Vector2 dir = to_obs.normalized();
                        beta.position = obs.position - dir * obs.radius;
                        float mu = obs.radius / dist;
                        beta.velocity = (agent.velocity - dir * agent.velocity.dot(dir)) * mu;
                    } else {
                        beta.position = obs.position + Vector2{obs.radius, 0};
                        beta.velocity = {0,0};
                    }
                }
                if (beta_agents.size() < max_beta_agents)
                    beta_agents.push_back(beta);
            }
        }
    }
}

void FlockSimulationCPU::prepare_spatial_hashing() {
    // Вычисляем хеш для каждого агента
    std::vector<int> hashes(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        float px = agents[i].position.x;
        float py = agents[i].position.y;
        int ix = (int)floorf((px + WORLD_BOUNDARY) / cell_size);
        int iy = (int)floorf((py + WORLD_BOUNDARY) / cell_size);
        ix = std::max(0, std::min(ix, grid_resolution - 1));
        iy = std::max(0, std::min(iy, grid_resolution - 1));
        hashes[i] = iy * grid_resolution + ix;
    }

    // Сортируем индексы агентов по хешу
    std::iota(agent_indices.begin(), agent_indices.end(), 0);
    std::sort(agent_indices.begin(), agent_indices.end(),
              [&](int a, int b) { return hashes[a] < hashes[b]; });

    // Сортируем самих агентов соответственно (требуется для последующего доступа по ячейкам)
    std::vector<Agent> sorted_agents(num_agents);
    for (int i = 0; i < num_agents; ++i)
        sorted_agents[i] = agents[agent_indices[i]];
    agents.swap(sorted_agents);

    // Пересчитываем хеши для отсортированного массива
    for (int i = 0; i < num_agents; ++i) {
        float px = agents[i].position.x;
        float py = agents[i].position.y;
        int ix = (int)floorf((px + WORLD_BOUNDARY) / cell_size);
        int iy = (int)floorf((py + WORLD_BOUNDARY) / cell_size);
        ix = std::max(0, std::min(ix, grid_resolution - 1));
        iy = std::max(0, std::min(iy, grid_resolution - 1));
        hashes[i] = iy * grid_resolution + ix;
    }

    // Заполняем границы ячеек
    std::fill(cell_start.begin(), cell_start.end(), -1);
    std::fill(cell_end.begin(), cell_end.end(), 0);
    for (int i = 0; i < num_agents; ++i) {
        int cell = hashes[i];
        if (cell_start[cell] == -1)
            cell_start[cell] = i;
        cell_end[cell] = i + 1;  // за последним
    }
}

void FlockSimulationCPU::compute_forces() {
    for (int i = 0; i < num_agents; ++i) {
        Agent& agent = agents[i];
        Vector2 alpha_force{0,0}, beta_force{0,0}, gamma_force{0,0};

        float px = agent.position.x;
        float py = agent.position.y;
        int ix = (int)floorf((px + WORLD_BOUNDARY) / cell_size);
        int iy = (int)floorf((py + WORLD_BOUNDARY) / cell_size);
        ix = std::max(0, std::min(ix, grid_resolution - 1));
        iy = std::max(0, std::min(iy, grid_resolution - 1));

        // альфа‑взаимодействия (по соседним ячейкам)
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = ix + dx, ny = iy + dy;
                if (nx < 0 || nx >= grid_resolution || ny < 0 || ny >= grid_resolution) continue;
                int cell = ny * grid_resolution + nx;
                int start = cell_start[cell];
                int end = cell_end[cell];
                if (start == -1) continue;
                for (int j = start; j < end; ++j) {
                    if (i == j) continue;
                    const Agent& other = agents[j];
                    Vector2 diff = other.position - agent.position;
                    float dist = diff.length();
                    if (dist < params.interaction_range && dist > 1e-6f) {
                        float z = sigma_norm(diff, params.epsilon);
                        Vector2 n_ij = sigma_epsilon(diff, params.epsilon);
                        alpha_force = alpha_force + n_ij * phi_alpha(z);
                        float a_ij = alpha_adjacency(agent.position, other.position);
                        alpha_force = alpha_force + (other.velocity - agent.velocity) * a_ij * (params.c2_alpha / params.c1_alpha);
                    }
                }
            }
        }
        alpha_force = alpha_force * params.c1_alpha;

        // бета‑взаимодействия
        for (const auto& beta : beta_agents) {
            Vector2 diff = beta.position - agent.position;
            float dist = diff.length();
            if (dist < params.obstacle_range && dist > 1e-6f) {
                float z = sigma_norm(diff, params.epsilon);
                Vector2 n_ik = sigma_epsilon(diff, params.epsilon);
                beta_force = beta_force + n_ik * phi_beta(z);
                float b_ik = beta_adjacency(agent.position, beta.position);
                beta_force = beta_force + (beta.velocity - agent.velocity) * b_ik * (params.c2_beta / params.c1_beta);
            }
        }
        beta_force = beta_force * params.c1_beta;

        // гамма‑цель
        if (params.use_gamma_target) {
            Vector2 diff = agent.position - params.gamma_target;
            float norm = diff.length();
            Vector2 pos_term = (norm < 1e-10f) ? Vector2{0,0} : diff * (1.0f / sqrtf(1.0f + norm * norm));
            Vector2 vel_term = agent.velocity - params.gamma_velocity;
            gamma_force = pos_term * (-params.c1_gamma) - vel_term * params.c2_gamma;
        }

        agent.acceleration = alpha_force + beta_force + gamma_force;
    }
}

void FlockSimulationCPU::integrate(float delta_time) {
    for (auto& a : agents) {
        a.velocity = a.velocity + a.acceleration * delta_time;
        float speed = a.velocity.length();
        const float max_speed = 100.0f;
        if (speed > max_speed) a.velocity = a.velocity.normalized() * max_speed;
        a.position = a.position + a.velocity * delta_time;

        const float boundary = WORLD_BOUNDARY;
        const float soft = boundary * 0.9f;
        if (fabsf(a.position.x) > soft) {
            float push = (boundary - fabsf(a.position.x)) / (boundary - soft);
            a.velocity.x += (a.position.x > 0 ? -1.0f : 1.0f) * push * 5.0f;
        }
        if (fabsf(a.position.y) > soft) {
            float push = (boundary - fabsf(a.position.y)) / (boundary - soft);
            a.velocity.y += (a.position.y > 0 ? -1.0f : 1.0f) * push * 5.0f;
        }
    }
}

void FlockSimulationCPU::step(float delta_time) {
    if (!running) return;
    generate_beta_agents();
    prepare_spatial_hashing();
    compute_forces();
    integrate(delta_time);
}

// ----- Заполнение VBO -----
void FlockSimulationCPU::build_agents_vbo(std::vector<Vertex>& verts) {
    verts.clear();
    verts.reserve(num_agents * 3);
    for (const auto& a : agents) {
        Vector2 dir = a.velocity.length() > 0.1f ? a.velocity.normalized() : Vector2{1, 0};
        Vector2 perp{-dir.y, dir.x};
        Vector2 tip = a.position + dir * 5.0f;
        Vector2 left = a.position - dir * 3.0f + perp * 3.0f;
        Vector2 right = a.position - dir * 3.0f - perp * 3.0f;
        verts.push_back({tip.x,   tip.y,   0.0f, 0.7f, 1.0f});
        verts.push_back({left.x,  left.y,  0.0f, 0.7f, 1.0f});
        verts.push_back({right.x, right.y, 0.0f, 0.7f, 1.0f});
    }
}

void FlockSimulationCPU::build_beta_vbo(std::vector<Vertex>& verts) {
    verts.clear();
    for (const auto& b : beta_agents) {
        float x = b.position.x, y = b.position.y;
        verts.push_back({x-2.0f, y-2.0f, 1.0f, 0.5f, 0.0f});
        verts.push_back({x+2.0f, y-2.0f, 1.0f, 0.5f, 0.0f});
        verts.push_back({x+2.0f, y+2.0f, 1.0f, 0.5f, 0.0f});
        verts.push_back({x-2.0f, y-2.0f, 1.0f, 0.5f, 0.0f});
        verts.push_back({x+2.0f, y+2.0f, 1.0f, 0.5f, 0.0f});
        verts.push_back({x-2.0f, y+2.0f, 1.0f, 0.5f, 0.0f});
    }
}

void FlockSimulationCPU::build_connections_vbo(std::vector<Vertex>& verts) {
    verts.clear();
    if (!show_connections) return;
    // Используем актуальную сетку (перестроенную в prepare_spatial_hashing)
    float alpha_range = params.interaction_range;
    float beta_range  = params.obstacle_range;

    for (int i = 0; i < num_agents; ++i) {
        const Agent& agent = agents[i];
        int ix = (int)floorf((agent.position.x + WORLD_BOUNDARY) / cell_size);
        int iy = (int)floorf((agent.position.y + WORLD_BOUNDARY) / cell_size);
        ix = std::max(0, std::min(ix, grid_resolution - 1));
        iy = std::max(0, std::min(iy, grid_resolution - 1));

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = ix + dx, ny = iy + dy;
                if (nx < 0 || nx >= grid_resolution || ny < 0 || ny >= grid_resolution) continue;
                int cell = ny * grid_resolution + nx;
                int start = cell_start[cell];
                int end = cell_end[cell];
                if (start == -1) continue;
                for (int j = start; j < end; ++j) {
                    if (i >= j) continue; // чтобы не дублировать рёбра
                    const Agent& other = agents[j];
                    Vector2 diff = other.position - agent.position;
                    float dist = diff.length();
                    if (dist < alpha_range) {
                        verts.push_back({agent.position.x, agent.position.y, 1.0f, 1.0f, 1.0f});
                        verts.push_back({other.position.x, other.position.y, 1.0f, 1.0f, 1.0f});
                        if (verts.size() >= max_connection_vertices) return;
                    }
                }
            }
        }

        // связи с β‑агентами
        for (const auto& beta : beta_agents) {
            Vector2 diff = beta.position - agent.position;
            float dist = diff.length();
            if (dist < beta_range) {
                verts.push_back({agent.position.x, agent.position.y, 1.0f, 0.5f, 0.0f});
                verts.push_back({beta.position.x, beta.position.y, 1.0f, 0.5f, 0.0f});
                if (verts.size() >= max_connection_vertices) return;
            }
        }
    }
}

void FlockSimulationCPU::fill_vbos() {
    std::vector<Vertex> verts;

    // Агенты
    build_agents_vbo(verts);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_agents);
    glBufferSubData(GL_ARRAY_BUFFER, 0, verts.size() * sizeof(Vertex), verts.data());

    // β-агенты
    if (show_beta_agents) {
        build_beta_vbo(verts);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_beta);
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.size() * sizeof(Vertex), verts.data());
    }

    // Связи
    if (show_connections) {
        build_connections_vbo(verts);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_connections);
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.size() * sizeof(Vertex), verts.data());
        h_connection_count = std::min((int)verts.size(), max_connection_vertices);
    } else {
        h_connection_count = 0;
    }
}

void FlockSimulationCPU::register_gl_buffers(GLuint agents, GLuint beta, GLuint conn) {
    vbo_agents = agents;
    vbo_beta = beta;
    vbo_connections = conn;
}
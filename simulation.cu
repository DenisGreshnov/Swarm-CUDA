#include "simulation.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <corecrt_math_defines.h>


// ----------------------------------------------------------------------
// Вспомогательные функции (σ-норма, bump, φ) для GPU
// ----------------------------------------------------------------------
__device__ double sigma_norm_device(const Vector2& z, double epsilon) {
    double n = z.length();
    return (1.0 / epsilon) * (sqrt(1.0 + epsilon * n * n) - 1.0);
}

__device__ Vector2 sigma_epsilon_device(const Vector2& z, double epsilon) {
    double n = z.length();
    if (n < 1e-10) return Vector2(0,0);
    return z * (1.0 / sqrt(1.0 + epsilon * n * n));
}

__device__ double bump_function_device(double z, double h) {
    if (z < h) return 1.0;
    if (z < 1.0) return 0.5 * (1.0 + cos(M_PI * (z - h) / (1.0 - h)));
    return 0.0;
}

__device__ double sigma1_device(double s) {
    return s / sqrt(1.0 + s * s);
}

__device__ double phi_alpha_device(double z, const SimParams& p) {
    double d_alpha = sigma_norm_device(Vector2(p.desired_distance, 0), p.epsilon);
    double r_alpha = sigma_norm_device(Vector2(p.interaction_range, 0), p.epsilon);
    double bump = bump_function_device(z / r_alpha, p.h_alpha);
    double s = z - d_alpha;
    double c = fabs(p.a - p.b) / sqrt(4.0 * p.a * p.b);
    double phi_s = 0.5 * ((p.a + p.b) * sigma1_device(s + c) + (p.a - p.b));
    return bump * phi_s;
}

__device__ double phi_beta_device(double z, const SimParams& p) {
    double d_beta = sigma_norm_device(Vector2(p.desired_distance * 0.6, 0), p.epsilon);
    double bump = bump_function_device(z / d_beta, p.h_beta);
    double s = z - d_beta;
    double action = sigma1_device(s) - 1.0;
    return bump * action;
}

__device__ double alpha_adjacency_device(const Vector2& qi, const Vector2& qj, const SimParams& p) {
    double dist = sigma_norm_device(qj - qi, p.epsilon);
    double r_alpha = sigma_norm_device(Vector2(p.interaction_range, 0), p.epsilon);
    return bump_function_device(dist / r_alpha, p.h_alpha);
}

__device__ double beta_adjacency_device(const Vector2& qi, const Vector2& qb, const SimParams& p) {
    double dist = sigma_norm_device(qb - qi, p.epsilon);
    double d_beta = sigma_norm_device(Vector2(p.desired_distance * 0.6, 0), p.epsilon);
    return bump_function_device(dist / d_beta, p.h_beta);
}

// ----------------------------------------------------------------------
// Ядро генерации β-агентов
// Каждый поток обрабатывает одну пару (агент, препятствие)
// ----------------------------------------------------------------------
__global__ void generate_beta_agents_kernel(
    const Agent* agents, int num_agents,
    const Obstacle* obstacles, int num_obstacles,
    BetaAgent* beta_agents, int* beta_counter,
    const SimParams params, int max_beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_agents * num_obstacles;
    if (idx >= total_pairs) return;

    int i = idx / num_obstacles;   // индекс агента
    int j = idx % num_obstacles;   // индекс препятствия

    const Agent& agent = agents[i];
    const Obstacle& obs = obstacles[j];

    Vector2 to_obs = obs.position - agent.position;
    double dist = to_obs.length();

    if (dist < params.obstacle_range + obs.radius) {
        BetaAgent beta;

        if (obs.is_wall) {
            // Упрощённая проекция на стену (как в исходном коде)
            if (fabs(obs.position.x - agent.position.x) < fabs(obs.position.y - agent.position.y)) {
                beta.position = Vector2(agent.position.x, obs.position.y);
                beta.velocity = Vector2(agent.velocity.x, 0);
            } else {
                beta.position = Vector2(obs.position.x, agent.position.y);
                beta.velocity = Vector2(0, agent.velocity.y);
            }
        } else {
            // Проекция на сферическое препятствие
            if (dist > 0.1) {
                Vector2 dir = to_obs.normalized();
                beta.position = obs.position - dir * obs.radius;
                double mu = obs.radius / dist;
                beta.velocity = (agent.velocity - dir * agent.velocity.dot(dir)) * mu;
            } else {
                beta.position = obs.position + Vector2(obs.radius, 0);
                beta.velocity = Vector2(0,0);
            }
        }

        // Атомарно добавляем в массив β-агентов
        int pos = atomicAdd(beta_counter, 1);
        if (pos < max_beta) {
            beta_agents[pos] = beta;
        } else {
            // Если переполнение, откатываем счётчик (необязательно, но для чистоты)
            atomicSub(beta_counter, 1);
        }
    }
}

// ----------------------------------------------------------------------
// Ядро вычисления сил для каждого агента
// ----------------------------------------------------------------------
__global__ void compute_forces_kernel(
    Agent* agents, int num_agents,
    const BetaAgent* beta_agents, int num_beta,
    const SimParams* params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    Agent& agent = agents[i];
    Vector2 alpha_force(0,0);
    Vector2 beta_force(0,0);
    Vector2 gamma_force(0,0);

    // --- α-сила (взаимодействие с другими агентами) ---
    for (int j = 0; j < num_agents; ++j) {
        if (i == j) continue;
        const Agent& other = agents[j];
        Vector2 diff = other.position - agent.position;
        double dist = diff.length();
        if (dist < params->interaction_range && dist > 0.1) {
            double z = sigma_norm_device(diff, params->epsilon);
            Vector2 n_ij = sigma_epsilon_device(diff, params->epsilon);
            alpha_force = alpha_force + n_ij * phi_alpha_device(z, *params);

            double a_ij = alpha_adjacency_device(agent.position, other.position, *params);
            alpha_force = alpha_force + (other.velocity - agent.velocity) * a_ij * (params->c2_alpha / params->c1_alpha);
        }
    }
    alpha_force = alpha_force * params->c1_alpha;

    // --- β-сила (отталкивание от β-агентов) ---
    for (int k = 0; k < num_beta; ++k) {
        const BetaAgent& beta = beta_agents[k];
        Vector2 diff = beta.position - agent.position;
        double dist = diff.length();
        if (dist < params->obstacle_range && dist > 0.1) {
            double z = sigma_norm_device(diff, params->epsilon);
            Vector2 n_ik = sigma_epsilon_device(diff, params->epsilon);
            beta_force = beta_force + n_ik * phi_beta_device(z, *params);

            double b_ik = beta_adjacency_device(agent.position, beta.position, *params);
            beta_force = beta_force + (beta.velocity - agent.velocity) * b_ik * (params->c2_beta / params->c1_beta);
        }
    }
    beta_force = beta_force * params->c1_beta;

    // --- γ-сила (навигация к цели) ---
    if (params->use_gamma_target) {
        Vector2 diff = agent.position - params->gamma_target;
        double norm = diff.length();
        Vector2 pos_term = (norm < 1e-10) ? Vector2(0,0) : diff * (1.0 / sqrt(1.0 + norm * norm));
        Vector2 vel_term = agent.velocity - params->gamma_velocity;
        gamma_force = pos_term * (-params->c1_gamma) - vel_term * params->c2_gamma;
    }

    agent.acceleration = alpha_force + beta_force + gamma_force;
}

// ----------------------------------------------------------------------
// Ядро интегрирования
// ----------------------------------------------------------------------
__global__ void integrate_kernel(Agent* agents, int num_agents, double delta_time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    Agent& a = agents[i];
    a.velocity = a.velocity + a.acceleration * delta_time;

    // Ограничение скорости
    double speed = a.velocity.length();
    const double max_speed = 400.0;
    if (speed > max_speed) {
        a.velocity = a.velocity.normalized() * max_speed;
    }

    a.position = a.position + a.velocity * delta_time;

    // Мягкие границы
    const double boundary = WORLD_BOUNDARY;
    const double soft = WORLD_BOUNDARY * 0.9;
    if (fabs(a.position.x) > soft) {
        double push = (boundary - fabs(a.position.x)) / (boundary - soft);
        a.velocity.x += (a.position.x > 0 ? -1.0 : 1.0) * push * 5.0;
    }
    if (fabs(a.position.y) > soft) {
        double push = (boundary - fabs(a.position.y)) / (boundary - soft);
        a.velocity.y += (a.position.y > 0 ? -1.0 : 1.0) * push * 5.0;
    }
}

// ----------------------------------------------------------------------
// Реализация методов класса FlockSimulation
// ----------------------------------------------------------------------
FlockSimulation::FlockSimulation() {
    // Параметры по умолчанию
    params.desired_distance   = 10.0;
    params.interaction_range  = 12.0;
    params.obstacle_range     = 8.0;
    params.c1_alpha = 8.0;  params.c2_alpha = 6.0;
    params.c1_beta  = 100.0;  params.c2_beta  = 10.0;
    params.c1_gamma = 7.0;  params.c2_gamma = 5.0;
    params.epsilon = 0.1;
    params.h_alpha = 0.2;  params.h_beta = 0.8;
    params.a = 5.0;        params.b = 7.0;
    params.use_gamma_target = true;
    params.gamma_target = Vector2(0,0);
    params.gamma_velocity = Vector2(0,0);

    // Инициализация агентов на CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-WORLD_BOUNDARY*0.75, WORLD_BOUNDARY*0.75);
    h_agents.resize(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        h_agents[i].position = Vector2(dis(gen), dis(gen));
        h_agents[i].velocity = Vector2(dis(gen)*0.05, dis(gen)*0.05);
        h_agents[i].acceleration = Vector2(0,0);
    }

    allocate_gpu_memory();
    copy_agents_to_gpu();
    sync_params_to_gpu();
}

FlockSimulation::~FlockSimulation() {
    free_gpu_memory();
}

void FlockSimulation::allocate_gpu_memory() {
    cudaMalloc(&d_agents, num_agents * sizeof(Agent));
    cudaMalloc(&d_obstacles, max_obstacles * sizeof(Obstacle));
    cudaMalloc(&d_beta_agents, max_beta_agents * sizeof(BetaAgent));
    cudaMalloc(&d_beta_count, sizeof(int));
    cudaMalloc(&d_params, sizeof(SimParams));
}

void FlockSimulation::free_gpu_memory() {
    cudaFree(d_agents);
    cudaFree(d_obstacles);
    cudaFree(d_beta_agents);
    cudaFree(d_beta_count);
    cudaFree(d_params);
}

void FlockSimulation::sync_params_to_gpu() {
    cudaMemcpy(d_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice);
}

void FlockSimulation::copy_agents_to_gpu() {
    cudaMemcpy(d_agents, h_agents.data(), num_agents * sizeof(Agent), cudaMemcpyHostToDevice);
}

void FlockSimulation::copy_obstacles_to_gpu() {
    cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);
}

void FlockSimulation::copy_beta_agents_from_gpu(int count) {
    h_beta_agents.resize(count);
    cudaMemcpy(h_beta_agents.data(), d_beta_agents, count * sizeof(BetaAgent), cudaMemcpyDeviceToHost);
    h_beta_count = count;
}

// Шаг симуляции (выполняется на GPU)
void FlockSimulation::step(double delta_time) {
    if (!running) return;

    // 1. Генерация β-агентов
    generate_beta_agents();

    // 2. Вычисление ускорений
    compute_forces();

    // 3. Интегрирование
    integrate(delta_time);

    // 4. Копирование результатов на CPU для рендеринга
    cudaMemcpy(h_agents.data(), d_agents, num_agents * sizeof(Agent), cudaMemcpyDeviceToHost);

    // Количество β-агентов уже получено в generate_beta_agents()
}

void FlockSimulation::generate_beta_agents() {
    // Обнуляем счётчик
    cudaMemset(d_beta_count, 0, sizeof(int));

    int total_pairs = num_agents * num_obstacles;
    if (total_pairs == 0) {
        h_beta_count = 0;
        return;
    }

    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    generate_beta_agents_kernel<<<blocks, threads>>>(
        d_agents, num_agents,
        d_obstacles, num_obstacles,
        d_beta_agents, d_beta_count,
        params, max_beta_agents
    );

    // Копируем счётчик и, если нужно, массив β-агентов
    int beta_count;
    cudaMemcpy(&beta_count, d_beta_count, sizeof(int), cudaMemcpyDeviceToHost);
    beta_count = std::min(beta_count, max_beta_agents);
    copy_beta_agents_from_gpu(beta_count);
}

void FlockSimulation::compute_forces() {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    compute_forces_kernel<<<blocks, threads>>>(
        d_agents, num_agents,
        d_beta_agents, h_beta_count,  // количество β-агентов уже известно
        d_params
    );
}

void FlockSimulation::integrate(double delta_time) {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    integrate_kernel<<<blocks, threads>>>(d_agents, num_agents, delta_time);
}

// Управление препятствиями
void FlockSimulation::add_obstacle(const Vector2& pos, double radius) {
    if (num_obstacles >= max_obstacles) return;
    h_obstacles.push_back({pos, radius, false, Vector2(0,0)});
    num_obstacles = h_obstacles.size();
    copy_obstacles_to_gpu();
}

void FlockSimulation::clear_obstacles() {
    h_obstacles.clear();
    num_obstacles = 0;
    copy_obstacles_to_gpu();
    // Сброс β-агентов
    h_beta_agents.clear();
    h_beta_count = 0;
    cudaMemset(d_beta_count, 0, sizeof(int));
}

void FlockSimulation::set_target(const Vector2& target) {
    params.gamma_target = target;
    params.use_gamma_target = true;
    sync_params_to_gpu();
}

// Геттеры для рендеринга (возвращают копии CPU)
std::vector<Agent> FlockSimulation::get_agents() const {
    return h_agents;
}

std::vector<Obstacle> FlockSimulation::get_obstacles() const {
    return h_obstacles;
}

std::vector<BetaAgent> FlockSimulation::get_beta_agents() const {
    return h_beta_agents;
}
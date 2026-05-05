#include "simulation.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <corecrt_math_defines.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// ----------------------------------------------------------------------
// Вспомогательные функции устройства
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
// Пространственное хэширование (вспомогательные ядра)
// ----------------------------------------------------------------------
__global__ void compute_hashes_kernel(
    const Agent* agents, int num_agents,
    int* hashes, double cell_size, double world_boundary, int grid_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    double px = agents[i].position.x;
    double py = agents[i].position.y;
    int ix = (int)floor((px + world_boundary) / cell_size);
    int iy = (int)floor((py + world_boundary) / cell_size);
    ix = max(0, min(ix, grid_res - 1));
    iy = max(0, min(iy, grid_res - 1));
    hashes[i] = iy * grid_res + ix;
}

__device__ int lower_bound_device(const int* arr, int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ int upper_bound_device(const int* arr, int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void build_cells_kernel(
    const int* sorted_hashes, int num_agents,
    int* cell_start, int* cell_end, int total_cells)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= total_cells) return;
    int start = lower_bound_device(sorted_hashes, num_agents, cell);
    int end = upper_bound_device(sorted_hashes, num_agents, cell);
    cell_start[cell] = start;
    cell_end[cell] = end;
}

// ----------------------------------------------------------------------
// Ядро генерации β-агентов (без изменений)
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
    int i = idx / num_obstacles;
    int j = idx % num_obstacles;
    const Agent& agent = agents[i];
    const Obstacle& obs = obstacles[j];
    Vector2 to_obs = obs.position - agent.position;
    double dist = to_obs.length();
    if (dist < params.obstacle_range + obs.radius) {
        BetaAgent beta;
        if (obs.is_wall) {
            if (fabs(obs.position.x - agent.position.x) < fabs(obs.position.y - agent.position.y)) {
                beta.position = Vector2(agent.position.x, obs.position.y);
                beta.velocity = Vector2(agent.velocity.x, 0);
            } else {
                beta.position = Vector2(obs.position.x, agent.position.y);
                beta.velocity = Vector2(0, agent.velocity.y);
            }
        } else {
            if (dist > 1e-6) {
                Vector2 dir = to_obs.normalized();
                beta.position = obs.position - dir * obs.radius;
                double mu = obs.radius / dist;
                beta.velocity = (agent.velocity - dir * agent.velocity.dot(dir)) * mu;
            } else {
                beta.position = obs.position + Vector2(obs.radius, 0);
                beta.velocity = Vector2(0,0);
            }
        }
        int pos = atomicAdd(beta_counter, 1);
        if (pos < max_beta) beta_agents[pos] = beta;
        else atomicSub(beta_counter, 1);
    }
}

// ----------------------------------------------------------------------
// Ядро вычисления сил (с пространственным хэшированием)
// ----------------------------------------------------------------------
__global__ void compute_forces_kernel(
    Agent* agents, int num_agents,
    const BetaAgent* beta_agents, int num_beta,
    const SimParams* params,
    const int* cell_start, const int* cell_end,
    int grid_res, double cell_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    Agent& agent = agents[i];
    Vector2 alpha_force(0,0), beta_force(0,0), gamma_force(0,0);

    double px = agent.position.x;
    double py = agent.position.y;
    int ix = (int)floor((px + WORLD_BOUNDARY) / cell_size);
    int iy = (int)floor((py + WORLD_BOUNDARY) / cell_size);
    ix = max(0, min(ix, grid_res - 1));
    iy = max(0, min(iy, grid_res - 1));

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = ix + dx, ny = iy + dy;
            if (nx < 0 || nx >= grid_res || ny < 0 || ny >= grid_res) continue;
            int cell = ny * grid_res + nx;
            int start = cell_start[cell], end = cell_end[cell];
            for (int j = start; j < end; ++j) {
                if (i == j) continue;
                const Agent& other = agents[j];
                Vector2 diff = other.position - agent.position;
                double dist = diff.length();
                if (dist < params->interaction_range && dist > 1e-6) {
                    double z = sigma_norm_device(diff, params->epsilon);
                    Vector2 n_ij = sigma_epsilon_device(diff, params->epsilon);
                    alpha_force = alpha_force + n_ij * phi_alpha_device(z, *params);
                    double a_ij = alpha_adjacency_device(agent.position, other.position, *params);
                    alpha_force = alpha_force + (other.velocity - agent.velocity) * a_ij * (params->c2_alpha / params->c1_alpha);
                }
            }
        }
    }
    alpha_force = alpha_force * params->c1_alpha;

    for (int k = 0; k < num_beta; ++k) {
        const BetaAgent& beta = beta_agents[k];
        Vector2 diff = beta.position - agent.position;
        double dist = diff.length();
        if (dist < params->obstacle_range && dist > 1e-6) {
            double z = sigma_norm_device(diff, params->epsilon);
            Vector2 n_ik = sigma_epsilon_device(diff, params->epsilon);
            beta_force = beta_force + n_ik * phi_beta_device(z, *params);
            double b_ik = beta_adjacency_device(agent.position, beta.position, *params);
            beta_force = beta_force + (beta.velocity - agent.velocity) * b_ik * (params->c2_beta / params->c1_beta);
        }
    }
    beta_force = beta_force * params->c1_beta;

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
// Ядра для заполнения VBO (геометрия агентов, β-агентов, связей)
// ----------------------------------------------------------------------
__global__ void build_agents_vbo_kernel(
    const Agent* agents, int num_agents,
    Vertex* vbo, int max_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    int base = i * 3;
    if (base + 2 >= max_vertices) return;

    const Agent& a = agents[i];
    Vector2 dir = a.velocity.length() > 0.1 ? a.velocity.normalized() : Vector2(1, 0);
    Vector2 perp(-dir.y, dir.x);
    Vector2 tip = a.position + dir * 5;
    Vector2 left = a.position - dir * 3 + perp * 3;
    Vector2 right = a.position - dir * 3 - perp * 3;

    vbo[base]   = { (float)tip.x, (float)tip.y, 0.0f, 0.7f, 1.0f };
    vbo[base+1] = { (float)left.x, (float)left.y, 0.0f, 0.7f, 1.0f };
    vbo[base+2] = { (float)right.x, (float)right.y, 0.0f, 0.7f, 1.0f };
}

__global__ void build_beta_vbo_kernel(
    const BetaAgent* beta_agents, int num_beta,
    Vertex* vbo, int max_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_beta) return;
    int base = i * 8;   // 2 треугольника (6 вершин) + 2 вершины линии
    if (base + 7 >= max_vertices) return;

    const BetaAgent& b = beta_agents[i];
    float x = (float)b.position.x, y = (float)b.position.y;

    // Квадрат (два треугольника)
    vbo[base]   = { x-2, y-2, 1.0f, 0.5f, 0.0f };
    vbo[base+1] = { x+2, y-2, 1.0f, 0.5f, 0.0f };
    vbo[base+2] = { x+2, y+2, 1.0f, 0.5f, 0.0f };
    vbo[base+3] = { x-2, y-2, 1.0f, 0.5f, 0.0f };
    vbo[base+4] = { x+2, y+2, 1.0f, 0.5f, 0.0f };
    vbo[base+5] = { x-2, y+2, 1.0f, 0.5f, 0.0f };

    // Линия направления
    if (b.velocity.length() > 0.5) {
        Vector2 dir = b.velocity.normalized();
        float ex = x + (float)dir.x * 6;
        float ey = y + (float)dir.y * 6;
        vbo[base+6] = { x, y, 1.0f, 0.5f, 0.0f };
        vbo[base+7] = { ex, ey, 1.0f, 0.5f, 0.0f };
    } else {
        vbo[base+6] = { x, y, 1.0f, 0.5f, 0.0f };
        vbo[base+7] = { x, y, 1.0f, 0.5f, 0.0f };
    }
}

__global__ void build_connections_vbo_kernel(
    const Agent* agents, int num_agents,
    const BetaAgent* beta_agents, int num_beta,
    const SimParams* params,
    const int* cell_start, const int* cell_end,
    int grid_res, double cell_size,
    Vertex* connections, int* conn_count,
    int max_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    const Agent& agent = agents[i];
    double alpha_range = params->interaction_range;
    double beta_range = params->obstacle_range;

    int ix = (int)floor((agent.position.x + WORLD_BOUNDARY) / cell_size);
    int iy = (int)floor((agent.position.y + WORLD_BOUNDARY) / cell_size);
    ix = max(0, min(ix, grid_res - 1));
    iy = max(0, min(iy, grid_res - 1));

    // α-α связи (белые) – только i < j
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = ix + dx, ny = iy + dy;
            if (nx < 0 || nx >= grid_res || ny < 0 || ny >= grid_res) continue;
            int cell = ny * grid_res + nx;
            int start = cell_start[cell], end = cell_end[cell];
            for (int j = start; j < end; ++j) {
                if (i >= j) continue;
                const Agent& other = agents[j];
                Vector2 diff = other.position - agent.position;
                double dist = diff.length();
                if (dist < alpha_range) {
                    int pos = atomicAdd(conn_count, 2);
                    if (pos + 1 < max_vertices) {
                        connections[pos]   = { (float)agent.position.x, (float)agent.position.y, 1.0f, 1.0f, 1.0f };
                        connections[pos+1] = { (float)other.position.x, (float)other.position.y, 1.0f, 1.0f, 1.0f };
                    } else {
                        atomicSub(conn_count, 2);
                    }
                }
            }
        }
    }

    // α-β связи (оранжевые)
    for (int k = 0; k < num_beta; ++k) {
        const BetaAgent& beta = beta_agents[k];
        Vector2 diff = beta.position - agent.position;
        double dist = diff.length();
        if (dist < beta_range) {
            int pos = atomicAdd(conn_count, 2);
            if (pos + 1 < max_vertices) {
                connections[pos]   = { (float)agent.position.x, (float)agent.position.y, 1.0f, 0.5f, 0.0f };
                connections[pos+1] = { (float)beta.position.x, (float)beta.position.y, 1.0f, 0.5f, 0.0f };
            } else {
                atomicSub(conn_count, 2);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Ядро интегрирования
// ----------------------------------------------------------------------
__global__ void integrate_kernel(Agent* agents, int num_agents, double delta_time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    Agent& a = agents[i];
    a.velocity = a.velocity + a.acceleration * delta_time;
    double speed = a.velocity.length();
    const double max_speed = 400.0;
    if (speed > max_speed) a.velocity = a.velocity.normalized() * max_speed;
    a.position = a.position + a.velocity * delta_time;

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

// ======================================================================
// Реализация FlockSimulation
// ======================================================================
FlockSimulation::FlockSimulation() {
    params.desired_distance   = 10.0;
    params.interaction_range  = 1.2 * params.desired_distance;
    params.obstacle_range     = 1.2 * 0.6 * params.desired_distance;
    params.c1_alpha = 50.0;  params.c2_alpha = 2.0;
    params.c1_beta  = 100.0; params.c2_beta  = 10.0;
    params.c1_gamma = 7.0;   params.c2_gamma = 0.5;
    params.epsilon = 0.1;
    params.h_alpha = 0.2;   params.h_beta = 0.9;
    params.a = 1.0;         params.b = 10.0;
    params.use_gamma_target = true;
    params.gamma_target = Vector2(0,0);
    params.gamma_velocity = Vector2(0,0);

    // Инициализация агентов на CPU (только для старта)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-WORLD_BOUNDARY*0.75, WORLD_BOUNDARY*0.75);
    h_agents_init.resize(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        h_agents_init[i].position = Vector2(dis(gen), dis(gen));
        h_agents_init[i].velocity = Vector2(dis(gen)/WORLD_BOUNDARY*15, dis(gen)/WORLD_BOUNDARY*15);
        h_agents_init[i].acceleration = Vector2(0,0);
    }

    cell_size = params.interaction_range * 1.05;
    grid_resolution = (int)ceil(2.0 * WORLD_BOUNDARY / cell_size) + 2;
    total_cells = grid_resolution * grid_resolution;

    max_connection_vertices = num_agents * 64;

    allocate_gpu_memory();
    copy_agents_to_gpu();   // однократная загрузка начальных данных
    sync_params_to_gpu();
}

FlockSimulation::~FlockSimulation() {
    unregister_gl_buffers();
    free_gpu_memory();
}

void FlockSimulation::allocate_gpu_memory() {
    cudaMalloc(&d_agents, num_agents * sizeof(Agent));
    cudaMalloc(&d_obstacles, max_obstacles * sizeof(Obstacle));
    cudaMalloc(&d_beta_agents, max_beta_agents * sizeof(BetaAgent));
    cudaMalloc(&d_beta_count, sizeof(int));
    cudaMalloc(&d_params, sizeof(SimParams));
    cudaMalloc(&d_hashes, num_agents * sizeof(int));
    cudaMalloc(&d_cell_start, total_cells * sizeof(int));
    cudaMalloc(&d_cell_end, total_cells * sizeof(int));
    cudaMalloc(&d_connection_count, sizeof(int));
}

void FlockSimulation::free_gpu_memory() {
    cudaFree(d_agents);
    cudaFree(d_obstacles);
    cudaFree(d_beta_agents);
    cudaFree(d_beta_count);
    cudaFree(d_params);
    cudaFree(d_hashes);
    cudaFree(d_cell_start);
    cudaFree(d_cell_end);
    cudaFree(d_connection_count);
}

void FlockSimulation::sync_params_to_gpu() {
    cudaMemcpy(d_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice);
}

void FlockSimulation::copy_agents_to_gpu() {
    cudaMemcpy(d_agents, h_agents_init.data(), num_agents * sizeof(Agent), cudaMemcpyHostToDevice);
    h_agents_init.clear();   // больше не нужны
}

void FlockSimulation::copy_obstacles_to_gpu() {
    cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);
}

// ----------------------------------------------------------------------
// CUDA-OpenGL interop
// ----------------------------------------------------------------------
void FlockSimulation::register_gl_buffers(GLuint vbo_agents, GLuint vbo_beta, GLuint vbo_connections) {
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_agents, vbo_agents, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_beta, vbo_beta, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_connections, vbo_connections, cudaGraphicsMapFlagsWriteDiscard);
}

void FlockSimulation::unregister_gl_buffers() {
    if (cuda_vbo_agents) cudaGraphicsUnregisterResource(cuda_vbo_agents);
    if (cuda_vbo_beta) cudaGraphicsUnregisterResource(cuda_vbo_beta);
    if (cuda_vbo_connections) cudaGraphicsUnregisterResource(cuda_vbo_connections);
    cuda_vbo_agents = cuda_vbo_beta = cuda_vbo_connections = nullptr;
}

void FlockSimulation::fill_vbos() {
    // --- Агенты ---
    if (cuda_vbo_agents) {
        cudaGraphicsMapResources(1, &cuda_vbo_agents);
        size_t num_bytes;
        Vertex* d_vbo_agents = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_agents, &num_bytes, cuda_vbo_agents);
        int max_vert_agents = num_bytes / sizeof(Vertex);
        int blocks = (num_agents + 255) / 256;
        build_agents_vbo_kernel<<<blocks, 256>>>(d_agents, num_agents, d_vbo_agents, max_vert_agents);
        cudaGraphicsUnmapResources(1, &cuda_vbo_agents);
    }

    // --- β-агенты ---
    if (cuda_vbo_beta && show_beta_agents) {
        cudaGraphicsMapResources(1, &cuda_vbo_beta);
        size_t num_bytes;
        Vertex* d_vbo_beta = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_beta, &num_bytes, cuda_vbo_beta);
        int max_vert_beta = num_bytes / sizeof(Vertex);
        int blocks = (h_beta_count + 255) / 256;
        build_beta_vbo_kernel<<<blocks, 256>>>(d_beta_agents, h_beta_count, d_vbo_beta, max_vert_beta);
        cudaGraphicsUnmapResources(1, &cuda_vbo_beta);
    }

    // --- Связи ---
    if (cuda_vbo_connections && show_connections) {
        cudaGraphicsMapResources(1, &cuda_vbo_connections);
        size_t num_bytes;
        Vertex* d_vbo_conn = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_conn, &num_bytes, cuda_vbo_connections);
        int max_vert_conn = num_bytes / sizeof(Vertex);

        cudaMemset(d_connection_count, 0, sizeof(int));
        int blocks = (num_agents + 255) / 256;
        build_connections_vbo_kernel<<<blocks, 256>>>(
            d_agents, num_agents,
            d_beta_agents, h_beta_count,
            d_params,
            d_cell_start, d_cell_end,
            grid_resolution, cell_size,
            d_vbo_conn, d_connection_count,
            max_vert_conn
        );
        cudaGraphicsUnmapResources(1, &cuda_vbo_connections);

        cudaMemcpy(&h_connection_count, d_connection_count, sizeof(int), cudaMemcpyDeviceToHost);
        h_connection_count = std::min(h_connection_count, max_connection_vertices);
    } else {
        h_connection_count = 0;
    }
}

// ----------------------------------------------------------------------
// Шаг симуляции (только физика)
// ----------------------------------------------------------------------
void FlockSimulation::step(double delta_time) {
    if (!running) return;

    generate_beta_agents();
    prepare_spatial_hashing();
    compute_forces();
    integrate(delta_time);
}

void FlockSimulation::generate_beta_agents() {
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
    cudaMemcpy(&h_beta_count, d_beta_count, sizeof(int), cudaMemcpyDeviceToHost);
    h_beta_count = std::min(h_beta_count, max_beta_agents);
    h_beta_agents.resize(h_beta_count);
    cudaMemcpy(h_beta_agents.data(), d_beta_agents, h_beta_count * sizeof(BetaAgent), cudaMemcpyDeviceToHost);
    // β-агенты нужны на CPU для следующего вызова compute_forces, поэтому копируем.
    // (копирование небольшое, альтернатива – хранить в unified memory, оставим так)
}

void FlockSimulation::prepare_spatial_hashing() {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    compute_hashes_kernel<<<blocks, threads>>>(d_agents, num_agents, d_hashes, cell_size, WORLD_BOUNDARY, grid_resolution);
    thrust::device_ptr<int> hash_ptr(d_hashes);
    thrust::device_ptr<Agent> agent_ptr(d_agents);
    thrust::sort_by_key(hash_ptr, hash_ptr + num_agents, agent_ptr);
    int cell_blocks = (total_cells + threads - 1) / threads;
    build_cells_kernel<<<cell_blocks, threads>>>(d_hashes, num_agents, d_cell_start, d_cell_end, total_cells);
}

void FlockSimulation::compute_forces() {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    compute_forces_kernel<<<blocks, threads>>>(d_agents, num_agents, d_beta_agents, h_beta_count, d_params,
                                                d_cell_start, d_cell_end, grid_resolution, cell_size);
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
    h_beta_agents.clear();
    h_beta_count = 0;
    cudaMemset(d_beta_count, 0, sizeof(int));
}

void FlockSimulation::set_target(const Vector2& target) {
    params.gamma_target = target;
    params.use_gamma_target = true;
    sync_params_to_gpu();
}
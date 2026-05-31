#include "simulation.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// ----------------------------------------------------------------------
// Вспомогательные функции устройства
// ----------------------------------------------------------------------
__constant__ SimParams c_params;

__device__ const float PI = 3.14159265358979323846f;

__device__ float sigma_norm_device(const Vector2& z, float epsilon) {
    float n = z.length();
    return (1.0f / epsilon) * (sqrtf(1.0f + epsilon * n * n) - 1.0f);
}

__device__ Vector2 sigma_epsilon_device(const Vector2& z, float epsilon) {
    float n = z.length();
    if (n < 1e-10f) return {0,0};
    return z * (1.0f / sqrtf(1.0f + epsilon * n * n));
}

__device__ float bump_function_device(float z, float h) {
    if (z < h) return 1.0f;
    if (z < 1.0f) return 0.5f * (1.0f + cosf(PI * (z - h) / (1.0f - h)));
    return 0.0f;
}

__device__ float sigma1_device(float s) {
    return s / sqrtf(1.0f + s * s);
}

__device__ float phi_alpha_device(float z, const SimParams& p) {
    float bump = bump_function_device(z / p.sigma_r_alpha, p.h_alpha);
    float s = z - p.sigma_d_alpha;
    float phi_s = 0.5f * ((p.a + p.b) * sigma1_device(s + p.c_phi) + (p.a - p.b));
    return bump * phi_s;
}

__device__ float phi_beta_device(float z, const SimParams& p) {
    float bump = bump_function_device(z / p.sigma_d_beta, p.h_beta);
    float s = z - p.sigma_d_beta;
    float action = sigma1_device(s) - 1.0f;
    return bump * action;
}

__device__ float alpha_adjacency_device(const Vector2& qi, const Vector2& qj, const SimParams& p) {
    float dist = sigma_norm_device(qj - qi, p.epsilon);
    return bump_function_device(dist / p.sigma_r_alpha, p.h_alpha);
}

__device__ float beta_adjacency_device(const Vector2& qi, const Vector2& qb, const SimParams& p) {
    float dist = sigma_norm_device(qb - qi, p.epsilon);
    return bump_function_device(dist / p.sigma_d_beta, p.h_beta);
}

// ----------------------------------------------------------------------
// Пространственное хэширование (вспомогательные ядра)
// ----------------------------------------------------------------------
__global__ void compute_hashes_kernel(
    const Agent* agents, int num_agents,
    int* hashes, float cell_size, float world_boundary, int grid_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    float px = agents[i].position.x;
    float py = agents[i].position.y;
    int ix = (int)floorf((px + world_boundary) / cell_size);
    int iy = (int)floorf((py + world_boundary) / cell_size);
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
// Ядро генерации β-агентов
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
        int pos = atomicAdd(beta_counter, 1);
        if (pos < max_beta) beta_agents[pos] = beta;
        else atomicSub(beta_counter, 1);
    }
}

// ----------------------------------------------------------------------
// Ядро вычисления сил
// ----------------------------------------------------------------------
__global__ void compute_forces_kernel(
    Agent* agents, int num_agents,
    const BetaAgent* beta_agents, const int* d_beta_count,
    const int* cell_start, const int* cell_end,
    int grid_res, float cell_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    Agent& agent = agents[i];
    Vector2 alpha_force{0,0}, beta_force{0,0}, gamma_force{0,0};

    float px = agent.position.x;
    float py = agent.position.y;
    int ix = (int)floorf((px + WORLD_BOUNDARY) / cell_size);
    int iy = (int)floorf((py + WORLD_BOUNDARY) / cell_size);
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
                float dist = diff.length();
                if (dist < c_params.interaction_range && dist > 1e-6f) {
                    float z = sigma_norm_device(diff, c_params.epsilon);
                    Vector2 n_ij = sigma_epsilon_device(diff, c_params.epsilon);
                    alpha_force = alpha_force + n_ij * phi_alpha_device(z, c_params);
                    float a_ij = alpha_adjacency_device(agent.position, other.position, c_params);
                    alpha_force = alpha_force + (other.velocity - agent.velocity) * a_ij * (c_params.c2_alpha / c_params.c1_alpha);
                }
            }
        }
    }
    alpha_force = alpha_force * c_params.c1_alpha;

    int num_beta = (d_beta_count) ? d_beta_count[0] : 0;
    for (int k = 0; k < num_beta; ++k) {
        const BetaAgent& beta = beta_agents[k];
        Vector2 diff = beta.position - agent.position;
        float dist = diff.length();
        if (dist < c_params.obstacle_range && dist > 1e-6f) {
            float z = sigma_norm_device(diff, c_params.epsilon);
            Vector2 n_ik = sigma_epsilon_device(diff, c_params.epsilon);
            beta_force = beta_force + n_ik * phi_beta_device(z, c_params);
            float b_ik = beta_adjacency_device(agent.position, beta.position, c_params);
            beta_force = beta_force + (beta.velocity - agent.velocity) * b_ik * (c_params.c2_beta / c_params.c1_beta);
        }
    }
    beta_force = beta_force * c_params.c1_beta;

    if (c_params.use_gamma_target) {
        Vector2 diff = agent.position - c_params.gamma_target;
        float norm = diff.length();
        Vector2 pos_term = (norm < 1e-10f) ? Vector2{0,0} : diff * (1.0f / sqrtf(1.0f + norm * norm));
        Vector2 vel_term = agent.velocity - c_params.gamma_velocity;
        gamma_force = pos_term * (-c_params.c1_gamma) - vel_term * c_params.c2_gamma;
    }
    agent.acceleration = alpha_force + beta_force + gamma_force;
}

// ----------------------------------------------------------------------
// Ядра для заполнения VBO
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
    Vector2 dir = a.velocity.length() > 0.1f ? a.velocity.normalized() : Vector2{1, 0};
    Vector2 perp{-dir.y, dir.x};
    Vector2 tip = a.position + dir * 5.0f;
    Vector2 left = a.position - dir * 3.0f + perp * 3.0f;
    Vector2 right = a.position - dir * 3.0f - perp * 3.0f;

    vbo[base]   = { tip.x,   tip.y,   0.0f, 0.7f, 1.0f };
    vbo[base+1] = { left.x,  left.y,  0.0f, 0.7f, 1.0f };
    vbo[base+2] = { right.x, right.y, 0.0f, 0.7f, 1.0f };
}

__global__ void build_beta_vbo_kernel(
    const BetaAgent* beta_agents, const int* d_beta_count,
    Vertex* vbo, int max_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_beta = d_beta_count[0];
    if (i >= num_beta) return;
    int base = i * 6;
    if (base + 5 >= max_vertices) return;

    const BetaAgent& b = beta_agents[i];
    float x = b.position.x, y = b.position.y;

    vbo[base]   = { x-2.0f, y-2.0f, 1.0f, 0.5f, 0.0f };
    vbo[base+1] = { x+2.0f, y-2.0f, 1.0f, 0.5f, 0.0f };
    vbo[base+2] = { x+2.0f, y+2.0f, 1.0f, 0.5f, 0.0f };
    vbo[base+3] = { x-2.0f, y-2.0f, 1.0f, 0.5f, 0.0f };
    vbo[base+4] = { x+2.0f, y+2.0f, 1.0f, 0.5f, 0.0f };
    vbo[base+5] = { x-2.0f, y+2.0f, 1.0f, 0.5f, 0.0f };
}

// ----------------------------------------------------------------------
// Ядро построения связей
// ----------------------------------------------------------------------
__global__ void build_connections_vbo_kernel(
    const Agent* agents, int num_agents,
    const BetaAgent* beta_agents, const int* d_beta_count,
    const int* cell_start, const int* cell_end,
    int grid_res, float cell_size,
    Vertex* connections, int* conn_count,
    int max_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    const Agent& agent = agents[i];
    float alpha_range = c_params.interaction_range;
    float beta_range  = c_params.obstacle_range;

    int ix = (int)floorf((agent.position.x + WORLD_BOUNDARY) / cell_size);
    int iy = (int)floorf((agent.position.y + WORLD_BOUNDARY) / cell_size);
    ix = max(0, min(ix, grid_res - 1));
    iy = max(0, min(iy, grid_res - 1));

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
                float dist = diff.length();
                if (dist < alpha_range) {
                    int pos = atomicAdd(conn_count, 2);
                    if (pos + 1 < max_vertices) {
                        connections[pos]   = { agent.position.x, agent.position.y, 1.0f, 1.0f, 1.0f };
                        connections[pos+1] = { other.position.x, other.position.y, 1.0f, 1.0f, 1.0f };
                    } else {
                        atomicSub(conn_count, 2);
                    }
                }
            }
        }
    }

    int num_beta = d_beta_count[0];
    for (int k = 0; k < num_beta; ++k) {
        const BetaAgent& beta = beta_agents[k];
        Vector2 diff = beta.position - agent.position;
        float dist = diff.length();
        if (dist < beta_range) {
            int pos = atomicAdd(conn_count, 2);
            if (pos + 1 < max_vertices) {
                connections[pos]   = { agent.position.x, agent.position.y, 1.0f, 0.5f, 0.0f };
                connections[pos+1] = { beta.position.x, beta.position.y, 1.0f, 0.5f, 0.0f };
            } else {
                atomicSub(conn_count, 2);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Ядро интегрирования
// ----------------------------------------------------------------------
__global__ void integrate_kernel(Agent* agents, int num_agents, float delta_time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    Agent& a = agents[i];
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

// ======================================================================
// Реализация FlockSimulation
// ======================================================================

FlockSimulation::FlockSimulation() {
    params.desired_distance   = 10.0f;
    params.interaction_range  = 1.2f * params.desired_distance;
    params.obstacle_range     = 1.2f * 0.6f * params.desired_distance;
    params.c1_alpha = 20.0f;  params.c2_alpha = 15.0f;
    params.c1_beta  = 50.0f; params.c2_beta  = 3.0f;
    params.c1_gamma = 1.0f;    params.c2_gamma = 0.2f;
    params.epsilon = 0.1f;
    params.h_alpha = 0.2f;     params.h_beta = 0.9f;
    params.a = 5.0f;           params.b = 5.0f;
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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis1(-sqrt(num_agents)*params.desired_distance*0.6f, sqrt(num_agents)*params.desired_distance*0.6f);
    std::uniform_real_distribution<float> dis2(-10, 10);
    h_agents_init.resize(num_agents);
    for (int i = 0; i < num_agents; ++i) {
        h_agents_init[i].position = {dis1(gen), dis1(gen)};
        h_agents_init[i].velocity = {dis2(gen), dis2(gen)};
        h_agents_init[i].acceleration = {0,0};
    }

    cell_size = params.interaction_range * 1.05f;
    grid_resolution = (int)ceilf(2.0f * WORLD_BOUNDARY / cell_size) + 2;
    total_cells = grid_resolution * grid_resolution;

    max_connection_vertices = num_agents * 10;
    max_blocks_agents = (num_agents + 255) / 256;

    allocate_gpu_memory();
    copy_agents_to_gpu();
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
    cudaMalloc(&d_hashes, num_agents * sizeof(int));
    cudaMalloc(&d_cell_start, total_cells * sizeof(int));
    cudaMalloc(&d_cell_end, total_cells * sizeof(int));
    cudaMalloc(&d_connection_count, sizeof(int));
    cudaMalloc(&d_block_conn_counts, max_blocks_agents * sizeof(int));
    cudaMalloc(&d_block_conn_offsets, max_blocks_agents * sizeof(int));
}

void FlockSimulation::free_gpu_memory() {
    cudaFree(d_agents);
    cudaFree(d_obstacles);
    cudaFree(d_beta_agents);
    cudaFree(d_beta_count);
    cudaFree(d_hashes);
    cudaFree(d_cell_start);
    cudaFree(d_cell_end);
    cudaFree(d_connection_count);
    cudaFree(d_block_conn_counts);
    cudaFree(d_block_conn_offsets);
}

void FlockSimulation::sync_params_to_gpu() {
    cudaMemcpyToSymbol(c_params, &params, sizeof(SimParams));
}

void FlockSimulation::copy_agents_to_gpu() {
    cudaMemcpy(d_agents, h_agents_init.data(), num_agents * sizeof(Agent), cudaMemcpyHostToDevice);
    h_agents_init.clear();
}

void FlockSimulation::copy_obstacles_to_gpu() {
    cudaMemcpy(d_obstacles, h_obstacles.data(), num_obstacles * sizeof(Obstacle), cudaMemcpyHostToDevice);
}

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

        // Защита от нулевого количества блоков
        if (h_beta_count > 0) {
            int blocks = (h_beta_count + 255) / 256;
            build_beta_vbo_kernel<<<blocks, 256>>>(d_beta_agents, d_beta_count, d_vbo_beta, max_vert_beta);
        }
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
            d_beta_agents, d_beta_count,
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

void FlockSimulation::step(float delta_time) {
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
}

void FlockSimulation::prepare_spatial_hashing() {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    compute_hashes_kernel<<<blocks, threads>>>(
        d_agents, num_agents, d_hashes,
        cell_size, WORLD_BOUNDARY, grid_resolution
    );

    thrust::device_ptr<int>   hash_ptr(d_hashes);
    thrust::device_ptr<Agent> agent_ptr(d_agents);
    thrust::sort_by_key(thrust::device, hash_ptr, hash_ptr + num_agents, agent_ptr);

    int cell_blocks = (total_cells + threads - 1) / threads;
    build_cells_kernel<<<cell_blocks, threads>>>(
        d_hashes, num_agents, d_cell_start, d_cell_end, total_cells
    );
}

void FlockSimulation::compute_forces() {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    compute_forces_kernel<<<blocks, threads>>>(d_agents, num_agents, d_beta_agents, d_beta_count,
                                                d_cell_start, d_cell_end, grid_resolution, cell_size);
}

void FlockSimulation::integrate(float delta_time) {
    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;
    integrate_kernel<<<blocks, threads>>>(d_agents, num_agents, delta_time);
}

void FlockSimulation::add_obstacle(const Vector2& pos, float radius) {
    if (num_obstacles >= max_obstacles) return;
    h_obstacles.push_back({pos, radius, false, {0,0}});
    num_obstacles = h_obstacles.size();
    copy_obstacles_to_gpu();
}

void FlockSimulation::clear_obstacles() {
    h_obstacles.clear();
    num_obstacles = 0;
    copy_obstacles_to_gpu();
    h_beta_count = 0;
    cudaMemset(d_beta_count, 0, sizeof(int));
}

void FlockSimulation::set_target(const Vector2& target) {
    params.gamma_target = target;
    params.use_gamma_target = true;
    sync_params_to_gpu();
}
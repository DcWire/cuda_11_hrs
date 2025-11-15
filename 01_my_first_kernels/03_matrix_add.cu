#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 
#define K 512
#define N 256
#define BLOCK_SIZE 32

// A is M x K, B is K x N, C is M x N

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            float sum = 0.0f;
            for(int l=0; l<k; l++) {
                sum += A[i * k + l] * B[n * l + j];
            }
            C[i * n + j] = sum; 
        }
    }
}


__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    // what row am I accessing and what column am I accessing?
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l=0; l<k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum; 
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for(int i=0; i<(rows * cols); i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(void) {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c; 
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c_cpu = (float*)malloc(size_c); 
    h_c_gpu = (float*)malloc(size_c);
    
    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Initialize dimensions 
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);

    // Warm up runs
    printf("Performing warm-up runs...\n");
    for(int i=0; i<3; i++) {
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

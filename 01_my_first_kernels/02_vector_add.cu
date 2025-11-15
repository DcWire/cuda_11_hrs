#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>

#define N 10000000 // 10 million vector size
// 1D Block Size
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

// CPU vector addition 
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }

}

// 1D vector addition
// CUDA kernel for vector addition 
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    // blockIdx.x and not y because the block has only an x dimension
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if (idx < (nx * ny * nz)) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void init_vector(float *vec, int n) {
    for(int i=0; i<n; i++) {
        vec[i] = (float)rand() / RAND_MAX; 
    }
}

// Calculate the time to measure performance
double get_time() {
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    h_a = (float *) malloc(size);
    h_b = (float *) malloc(size);
    h_c_cpu = (float *) malloc(size);
    h_c_gpu = (float *) malloc(size);
    h_c_gpu_3d = (float *) malloc(size);
    
    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);
    
    // Allocate memory on the GPU
    // Address of the pointer
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Need to define the grid and block dimensions for 3d 
    int nx = 100, ny = 100, nz = 1000; // N = nx * ny * nz
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
            (nx + block_size_3d.x - 1) / block_size_3d.x, 
            (ny + block_size_3d.y - 1) / block_size_3d.y,
            (nz + block_size_3d.z - 1) / block_size_3d.z
    );


    printf("Performing warmup runs for cahce...\n");
    for (int i=0; i<3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
    }
    
    int runs = 20;
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i=0; i<runs; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time; 
    }

    double cpu_avg_time = cpu_total_time / (float)runs; 

    printf("Benchmarking 1D GPU implementation...\n");
    double gpu_total_time = 0.0;
    for(int i=0; i<runs; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    double gpu_avg_time = gpu_total_time / (float)runs; 
    
    printf("Benchmarking 3D GPU implementation...\n"); 
    
    double gpu_3d_total_time = 0.0;
    for(int i=0; i<runs; i++) {
        double start_time = get_time(); 
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    
    double gpu_avg_3d_time = gpu_3d_total_time / (float)runs;
    
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("GPU 3D AVG TIME: %f milliseonds \n", gpu_avg_3d_time*1000);
    printf("Speedup 1D: %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup 3D: %fx\n", cpu_avg_time / gpu_avg_3d_time);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_gpu_3d, d_c, size, cudaMemcpyDeviceToHost);

    bool correct = true; 
    for(int i=0; i<N; i++) {
        // Check against a threshold value
        if(fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    for(int i=0; i<N; i++) {
        // Check against a threshold value
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}

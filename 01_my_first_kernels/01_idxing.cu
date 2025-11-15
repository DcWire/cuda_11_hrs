/*
 * This file consists of my initial learnings about how indexes are accessed in a CUDA kernel
 * based on the video (check readme) and the nvidia samples
*/

#include <stdio.h>

// The goal is to calculate a unique id for every thread. The calculation MUST CHANGE based on the dimensionality of 
// the kernel launch, which in turn should match the dimensionality of your data. 
__global__ void whoami(void) {
    // So think of this in this way... a block within another block and there are gridDim.x blocks in the x direction
    // gridDim.y blocks in the y direction. For row major, its the x dimension + number of blocks in the x direction times 
    // y because that many x blocks occur for every y, and then same thing in the z direction by there would be x*y times 
    // blocks for every z
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)
    
    // Now this is similar to the previous calculation. Since we have the block id, we now calculate the block offset to get the
    // thread id. How many total people have passed -> How many total threads have passed for me to get here. This is 
    // in the threads domain. 
    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)
    
    // Now we calculate the thread offset. Meaning, the actual person inside the block. 
    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;
    
    // The thread id would be calculated from this offset 
    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;
    // Warp is basically the number of threads executed together. Something related to SIMT: Single Instruction Multiple Thread
    // Warp is a bunch of threads that are run together in parallel
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}

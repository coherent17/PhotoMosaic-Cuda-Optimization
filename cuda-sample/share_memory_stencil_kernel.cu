#include <stdio.h>

#define N 10
#define BLOCK_SIZE 5
#define THREADS_PER_BLOCK 9
#define RADIUS 2

__global__ void stencil_1d(int *in, int *out){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x + RADIUS;

    // Initialize shared memory with zeros
    if (threadIdx.x < BLOCK_SIZE + 2 * RADIUS) {
        temp[threadIdx.x] = 0;
    }

    // Synchronize to ensure all initializations are done
    __syncthreads();

    // Read input elements into shared memory
    if (global_index < N) {
        temp[local_index] = in[global_index];
    }

    if (threadIdx.x < RADIUS) {
        // Load Halo with certain thread, checking bounds
        if (global_index >= RADIUS) {
            temp[local_index - RADIUS] = in[global_index - RADIUS];
        } else {
            temp[local_index - RADIUS] = 0; // Handle boundary condition
        }

        if (global_index + BLOCK_SIZE < N) {
            temp[local_index + BLOCK_SIZE] = in[global_index + BLOCK_SIZE];
        } else {
            temp[local_index + BLOCK_SIZE] = 0; // Handle boundary condition
        }
    }

    // Synchronize (ensure all the data is available), avoid data race
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[local_index + offset];
    }

    // Store the result
    if (global_index < N) {
        out[global_index] = result;
    }
}

int main(){
    int *in, *out;
    int *d_in, *d_out;
    size_t size = N * sizeof(int);

    // Allocate host memory
    in = (int *)malloc(size);
    out = (int *)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        in[i] = 1;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy input data to the device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<numBlocks, THREADS_PER_BLOCK>>>(d_in, d_out);

    // Copy output from device to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Print output
    for (int i = 0; i < N; i++) {
        printf("out[%d] = %d\n", i, out[i]);
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Free host memory
    free(in);
    free(out);

    return 0;
}

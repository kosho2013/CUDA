#include <stdio.h>

__global__ void atomicAddSharedKernel(int *result) {
    // Declare shared memory
    __shared__ int shared_sum;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        shared_sum = 0;
    }

    // Synchronize to ensure shared memory initialization is complete
    __syncthreads();

    // Each thread adds 1 to the shared memory variable atomically
    atomicCAS(&shared_sum, 0, threadIdx.x);

    // Synchronize again to ensure all atomicAdd operations are complete
    __syncthreads();

    // The first thread writes the shared memory value to the result
    if (threadIdx.x == 0) {
        *result = shared_sum;
    }
}

int main() {
    int *d_result;
    int h_result;

    // Allocate device memory for result
    cudaMalloc((void**)&d_result, sizeof(int));

    // Launch kernel with 1 block and 256 threads
    atomicAddSharedKernel<<<1, 256>>>(d_result);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result: %d\n", h_result);

    // Free device memory
    cudaFree(d_result);


    printf("Size of size_t: %zu bytes\n", sizeof(size_t));

    return 0;
}

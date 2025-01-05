#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define MASK 5
#define RADIUS 2
#define IMAGE 1000
#define TILE 20

// -------------------------------------
// 1) GPU kernel for shared-memory 2D convolution
// -------------------------------------
__global__ void kernel(float *in, float *filter, float *out)
{
    int x_out = blockIdx.x * TILE + threadIdx.x;
    int y_out = blockIdx.y * TILE + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x_in = x_out - RADIUS;
    int y_in = y_out - RADIUS;

    __shared__ float shared_mem[TILE + MASK - 1][TILE + MASK - 1];

    // Load input tile + boundary "halo" into shared memory
    if (0 <= x_in && x_in < IMAGE && 0 <= y_in && y_in < IMAGE)
        shared_mem[ty][tx] = in[y_in * IMAGE + x_in];
    else
        shared_mem[ty][tx] = 0.0f;

    __syncthreads();

    // Only the interior threads perform the convolution
    if (tx < TILE && ty < TILE)
    {
        float value = 0.0f;
        for (int i = 0; i < MASK; i++)
        {
            for (int j = 0; j < MASK; j++)
            {
                value += filter[i * MASK + j] * shared_mem[ty + i][tx + j];
            }
        }
        // Write to global memory if within image bounds
        if (x_out < IMAGE && y_out < IMAGE)
        {
            out[y_out * IMAGE + x_out] = value;
        }
    }
}

// -------------------------------------
// 2) CPU reference implementation
// -------------------------------------
void cpuConvolution(const float* in, const float* filter, float* out,
                    int imageSize, int maskSize, int radius)
{
    for (int row = 0; row < imageSize; row++)
    {
        for (int col = 0; col < imageSize; col++)
        {
            float value = 0.0f;
            // For each pixel (row, col), apply the 5x5 filter
            for (int i = 0; i < maskSize; i++)
            {
                for (int j = 0; j < maskSize; j++)
                {
                    // Position in the input image
                    int x = col + j - radius;
                    int y = row + i - radius;

                    // Boundary check
                    if (x >= 0 && x < imageSize &&
                        y >= 0 && y < imageSize)
                    {
                        value += in[y * imageSize + x] * filter[i * maskSize + j];
                    }
                }
            }
            out[row * imageSize + col] = value;
        }
    }
}

// -------------------------------------
// 3) Main function
// -------------------------------------
int main(int argc, char **argv)
{
    size_t size_in     = IMAGE * IMAGE * sizeof(float);
    size_t size_filter = MASK * MASK * sizeof(float);
    size_t size_out    = IMAGE * IMAGE * sizeof(float);

    // Allocate and initialize host memory
    float *h_in     = (float *)malloc(size_in);
    float *h_filter = (float *)malloc(size_filter);
    float *h_out    = (float *)malloc(size_out);

    // Initialize values
    for (int i = 0; i < IMAGE * IMAGE; i++)
    {
        h_in[i] = 15.0f;
    }
    for (int i = 0; i < MASK * MASK; i++)
    {
        h_filter[i] = 5.0f;
    }

    // Allocate device memory
    float *d_in, *d_filter, *d_out;
    cudaMalloc((void **)&d_in,     size_in);
    cudaMalloc((void **)&d_filter, size_filter);
    cudaMalloc((void **)&d_out,    size_out);

    // Copy data to device
    cudaMemcpy(d_in,     h_in,     size_in,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(TILE + MASK - 1, TILE + MASK - 1);
    dim3 gridDim((IMAGE + TILE - 1) / TILE, (IMAGE + TILE - 1) / TILE);

    // Timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<gridDim, blockDim>>>(d_in, d_filter, d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    // Print performance info
    std::cout << "blockDim: (" << blockDim.x << ", "
              << blockDim.y << ", " << blockDim.z << ")" << std::endl;
    std::cout << "gridDim:  (" << gridDim.x << ", "
              << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "GPU Elapsed time: " << milliseconds << " ms" << std::endl;

    // -------------------------------------
    // Compute the reference convolution on the CPU
    // -------------------------------------
    float *gold = (float *)malloc(size_out);
    cpuConvolution(h_in, h_filter, gold, IMAGE, MASK, RADIUS);

    // Compare CPU and GPU results
    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < IMAGE * IMAGE; i++)
    {
        float err = fabs(gold[i] - h_out[i]);
        if (err > 1)
        {
          cout << gold[i] << " " << h_out[i] << endl;
        }
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_filter);
    cudaFree(d_out);

    // Free host memory
    free(h_in);
    free(h_filter);
    free(h_out);
    free(gold);

    return 0;
}

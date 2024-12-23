#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define TILE_SIZE 1024
#define STEPS 11

__global__ void vecAdd(float *in, float *out)
{
    int start = blockIdx.x * TILE_SIZE * 2;
    int tx = threadIdx.x;

    __shared__ float tile[TILE_SIZE * 2];

    // loading
    tile[tx] = in[start + tx];
    tile[tx + TILE_SIZE] = in[start + tx + TILE_SIZE];

    __syncthreads();
    
    int stride = TILE_SIZE;
    for (int i = 0; i < STEPS; i++)
    {
      if (tx < stride)
      {
        tile[tx] = tile[tx] + tile[tx + stride];
      }
      if (i == 0)
      {
        printf("%d, %f\n", tx, tile[tx]);
      }
      __syncthreads();
      stride /= 2;
    }

    if (tx == 0)
    {
      out[0] = tile[0];
    }
    
}

int main(int argc, char **argv) {
  size_t size_in = 2 * TILE_SIZE * sizeof(float);
  size_t size_out = sizeof(float);

  // Allocate and initialize host memory
  float *h_in = (float *)malloc(2 * size_in);
  float *h_out = (float *)malloc(size_out);

  for (int i = 0; i < 2 * TILE_SIZE; i++)
  {
    h_in[i] = 15.0f; 
  }

  // Allocate device memory
  float *d_in, *d_out;
  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_out, size_out);

  // Copy data from host to device
  cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

  // Launch the kernel
  dim3 blockDim(TILE_SIZE);
  dim3 gridDim(1);

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);

  vecAdd<<<gridDim, blockDim>>>(d_in, d_out);

  // Record the stop event
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy result back to host
  cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);


  // Print the elapsed time
  std::cout << "blockDim: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
  std::cout << "gridDim: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;
  


  float *gold = (float *)malloc(size_out);
  float value = 0.0f;
  for (int i = 0; i < 2 * TILE_SIZE; i++)
  {
    value += h_in[i];
  }
  gold[0] = value;

  cout << h_out[0] << endl;
  cout << gold[0] << endl;

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  // Free host memory
  free(h_in);
  free(h_out);

  return 0;
}

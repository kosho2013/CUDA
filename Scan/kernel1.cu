#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define TILE 1024
#define STEPS 10

__global__ void kernel(float *in, float *out)
{
    int bx = blockIdx.x * TILE;
    int tx = threadIdx.x;

    __shared__ float tile[TILE];

    // loading
    tile[tx] = in[bx + tx];
    __syncthreads();
    
    // computation
    int stride = TILE / 2;
    for (int i = 0; i < STEPS; i++)
    {
      if (tx < stride)
      {
        float tmp = tile[tx] + tile[tx + stride];
        __syncthreads();
        tile[tx] = tmp;
        __syncthreads();
      }
      stride /= 2;
    }

    if (tx == 0)
    {
      out[0] = tile[0];
    }
}



int main(int argc, char **argv) {
  size_t size_in = TILE * sizeof(float);
  size_t size_out = 1 * sizeof(float);

  // Allocate and initialize host memory
  float *h_in = (float *)malloc(2 * size_in);
  float *h_out = (float *)malloc(size_out);

  for (int i = 0; i < TILE; i++)
  {
    h_in[i] = i; 
  }

  // Allocate device memory
  float *d_in, *d_out;
  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_out, size_out);

  // Copy data from host to device
  cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

  // Launch the kernel
  dim3 blockDim(TILE);
  dim3 gridDim(1);

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);

  kernel<<<gridDim, blockDim>>>(d_in, d_out);

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
  


  cout << h_out[0] << endl;

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  // Free host memory
  free(h_in);
  free(h_out);

  return 0;
}

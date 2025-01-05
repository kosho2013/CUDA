#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>


using namespace std;

#define TILE 1024
#define NUM_INPUT 20000
#define NUM_BIN 256

__global__ void kernel(unsigned char *in, int *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ int tile[NUM_BIN];

    if (tx < NUM_BIN)
    {
      tile[tx] = 0;
    }
    __syncthreads();


    if (idx < NUM_INPUT)
    {
      atomicAdd(&tile[in[idx]], 1);
    }

    __syncthreads();

    if (tx < NUM_BIN)
    {
      atomicAdd(&out[tx], tile[tx]);
    }
}

int main(int argc, char **argv) {

  // Create a random device to seed the random number generator
  std::random_device rd;

  // Initialize the Mersenne Twister random number generator
  std::mt19937 gen(rd());

  // Define a uniform distribution from 0 to 255
  std::uniform_int_distribution<> dist(0, 255);


  size_t size_in = NUM_INPUT * sizeof(unsigned char);
  size_t size_out = NUM_BIN * sizeof(int);

  // Allocate and initialize host memory
  unsigned char *h_in = (unsigned char *)malloc(size_in);
  int *h_out = (int *)malloc(size_out);

  for (int i = 0; i < NUM_INPUT; i++)
  {
    h_in[i] = (unsigned char)(dist(gen)); 
  }

  for (int i = 0; i < NUM_BIN; i++)
  {
    h_out[i] = 0; 
  }

  // Allocate device memory
  unsigned char *d_in;
  int *d_out;
  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_out, size_out);

  // Copy data from host to device
  cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out, size_out, cudaMemcpyHostToDevice);

  

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);


  // Launch the kernel
  dim3 blockDim(TILE);
  dim3 gridDim((NUM_INPUT + TILE - 1) / TILE);
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
  


  int *gold = (int *)malloc(size_out);
  for (int i = 0; i < NUM_BIN; i++)
  {
    gold[i] = 0;
  }

  for (int i = 0; i < NUM_INPUT; i++)
  {
    gold[h_in[i]] += 1;
  }

  for (int i = 0; i < NUM_BIN; i++)
  {
    float err = fabs(gold[i] - h_out[i]);
    if (err > 0.01)
    {
      cout << i << " " << gold[i] << " " << h_out[i] << endl;
    }
    // cout << err << endl;
  }

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  // Free host memory
  free(h_in);
  free(h_out);

  return 0;
}



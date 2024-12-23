#include <cuda.h>
#include <iostream>
#include <cstdlib>

using namespace std;


#define TILE 1024

__global__ void kernel(float *A, float *B, float *C, const int m, const int k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float B_tile[TILE];


    int steps = (k + TILE-1) / TILE;
    float value = 0.0f;
    for (int i = 0; i < steps; i++)
    {

        if (i*TILE + tx < k)
        {
            B_tile[tx] = B[i*TILE + tx];
        } else
        {
            B_tile[tx] = 0.0f;
        }

        __syncthreads();
        
        for (int j = 0; j < TILE; j++)
        {
          if (x < m && i*TILE + j < k)
          {
            value += A[x * k + i*TILE + j] * B_tile[j];
          }
        }

        __syncthreads();
    }

    if (x < m)
    {
        C[x] = value;
    }
}


int main(int argc, char **argv) {
  int m = 10000;
  int k = 10000;
  int n = 1;

  size_t A_size = m * k * sizeof(float);
  size_t B_size = k * n * sizeof(float);
  size_t C_size = m * n * sizeof(float);
  
  // Allocate and initialize host memory
  float *h_A = (float *)malloc(A_size);
  float *h_B = (float *)malloc(B_size);
  float *h_C = (float *)malloc(C_size);

  for (int i = 0; i < m * k; i++) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < k * n; i++) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }



  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, A_size);
  cudaMalloc((void **)&d_B, B_size);
  cudaMalloc((void **)&d_C, C_size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
  

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);


  dim3 blockDim(TILE);
  int tmp = (m + TILE - 1) / TILE;
  dim3 gridDim(tmp);
  kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k);

  // Record the stop event
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost);


  // Print the elapsed time
  std::cout << "blockDim: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
  std::cout << "gridDim: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;
  
  float *gold = (float *)malloc(C_size);
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      float value = 0.0f;
      for (int v = 0; v < k; v++)
      {
        value += h_A[i*k+v] * h_B[v*n+j];
      }
      gold[i*n+j] = value;
    }
  }


  for (int i = 0; i < m*n; i++)
  {
      float err = fabs(gold[i] - h_C[i]);
      if (err > 0.01)
      {
        cout << err << endl;
      }
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

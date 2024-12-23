#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void vecAdd(float *A, float *B, float *C, int m, int k, int n) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < m && y < n)
  {
    float value = 0.0f;
    for (int i = 0; i < k; ++i)
    {
      value += A[x*k + i] * B[i*n + y];
    }
    C[x*n + y] = value;
  }
}

int main(int argc, char **argv) {
  int m = 100;
  int n = 200;
  int k = 300;

  size_t A_size = m * k * sizeof(float);
  size_t B_size = k * n * sizeof(float);
  size_t C_size = m * n * sizeof(float);
  
  // Allocate and initialize host memory
  float *h_A = (float *)malloc(A_size);
  float *h_B = (float *)malloc(B_size);
  float *h_C = (float *)malloc(C_size);

  for (int i = 0; i < m*k; i++) {
    h_A[i] = 5.0f;
  }

  for (int i = 0; i < k*n; i++) {
    h_B[i] = 5.0f;
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, A_size);
  cudaMalloc((void **)&d_B, B_size);
  cudaMalloc((void **)&d_C, C_size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);

  // Launch the kernel
  int block_1 = 32;
  int block_2 = 32;
  int numBlock_1 = (m + block_1 - 1) / block_1;
  int numBlock_2 = (n + block_2 - 1) / block_2;

  dim3 blockDim(block_1, block_2);
  dim3 gridDim(numBlock_1, numBlock_2);

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);

  vecAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);

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

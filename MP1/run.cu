#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

int main(int argc, char **argv) {
  int len = 1000000;
  size_t size = len * sizeof(float);

  // Allocate and initialize host memory
  float *h_in1 = (float *)malloc(size);
  float *h_in2 = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  for (int i = 0; i < len; i++) {
    h_in1[i] = 5.0f;
    h_in2[i] = 5.0f;
  }

  // Allocate device memory
  float *d_in1, *d_in2, *d_out;
  cudaMalloc((void **)&d_in1, size);
  cudaMalloc((void **)&d_in2, size);
  cudaMalloc((void **)&d_out, size);

  // Copy data from host to device
  cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice);

  // Launch the kernel
  int blockSize = 1024;
  int numBlocks = (len + blockSize - 1) / blockSize;

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);

  vecAdd<<<numBlocks, blockSize>>>(d_in1, d_in2, d_out, len);

  // Record the stop event
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy result back to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // Print the elapsed time
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  // Print the output values
//   for (int i = 0; i < len; i++) {
//     std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
//   }

  cout << "result:" << h_out[9999] << endl;

  // Free device memory
  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);

  // Free host memory
  free(h_in1);
  free(h_in2);
  free(h_out);

  return 0;
}

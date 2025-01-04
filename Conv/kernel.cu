#include <cuda.h>
#include <iostream>
#include <cstdlib>

using namespace std;

#define TILE 32
#define WIDTH 1000
#define MASK_WIDTH 5


__global__ void kernel(float *in, float *filter, float *out)
{
    int x = blockIdx.x * (TILE_SIZE - PAD) + threadIdx.x;
    int y = blockIdx.y * (TILE_SIZE - PAD) + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Load data into shared memory with proper boundary check
    if (x < n && y < n)
    {
        tile[tx][ty] = in[x * n + y];
    }
    else
    {
        tile[tx][ty] = 0.0f;
    }

    __syncthreads();

    // Compute output value for the valid region
    if (tx < TILE_SIZE - PAD && ty < TILE_SIZE - PAD)
    {
        if (x < n && y < n)
        {
            float value = 0.0f;
            for (int i = 0; i < MASK_WIDTH; i++)
            {
                for (int j = 0; j < MASK_WIDTH; j++)
                {
                    value += filter[i * MASK_WIDTH + j] * tile[tx + i][ty + j];
                }
            }

            out[x * n + y] = value;
        }
    }
}

int main(int argc, char **argv) {
  size_t size_in = WIDTH * WIDTH * sizeof(float);
  size_t size_filter = MASK_WIDTH * MASK_WIDTH * sizeof(float);
  size_t size_out = WIDTH * WIDTH * sizeof(float);

  // Allocate and initialize host memory
  float *h_in = (float *)malloc(size_in);
  float *h_filter = (float *)malloc(size_filter);
  float *h_out = (float *)malloc(size_out);

  for (int i = 0; i < n * n; i++)
  {
    h_in[i] = 15.0f;
  }

  for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
  {
    h_filter[i] = 5.0f;
  }

  // Allocate device memory
  float *d_in, *d_filter, *d_out;
  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_filter, size_filter);
  cudaMalloc((void **)&d_out, size_out);

  // Copy data from host to device
  cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

  // Launch the kernel
  int tmp = TILE_SIZE - PAD;
  int numBlock_1 = (n + tmp - 1) / tmp;
  int numBlock_2 = (n + tmp - 1) / tmp;

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim(numBlock_1, numBlock_2);

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);

  kernel<<<gridDim, blockDim>>>(d_in, d_filter, d_out, n);

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
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      float value = 0.0f;
      for (int v = 0; v < MASK_WIDTH; v++)
      {
        for (int u = 0; u < MASK_WIDTH; u++)
        {
          if (i+v < n && j+u < n)
          {
            value += h_in[(i+v)*n+j+u] * h_filter[v*MASK_WIDTH+u];
          }
        }
      }
      gold[i*n+j] = value;
    }
  }



  for (int i = 0; i < n*n; i++)
  {
      float err = fabs(gold[i] - h_out[i]);
      if (err > 0.01)
      {
        cout << i/n << "," << i%n << " " << gold[i] << " " << h_out[i] << endl;
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

  return 0;
}

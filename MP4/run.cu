#include <cuda.h>
#include <iostream>
#include <cstdlib>

using namespace std;

#define TILE 32
#define IN_WIDTH 1000
#define MASK_WIDTH 5
#define OUT_WIDTH 996


__global__ void kernel(float *in, float *filter, float *out)
{
    int x_in = blockIdx.x * blockDim.x + threadIdx.x;
    int y_in = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_out = x - MASK_WIDTH + 1;
    int y_out = y - MASK_WIDTH + 1;

    __shared__ float in_tile[TILE][TILE];

    // Load data into shared memory with proper boundary check
    if (x_in < WIDTH && y_in < WIDTH)
    {
        in_tile[ty][tx] = in[y_in * WIDTH + x_in];
    }
    else
    {
        in_tile[ty][tx] = 0.0f;
    }

    __syncthreads();


    if (x_out >= 0 && x_out < WIDTH-MASK_WIDTH+1 && y_out >= 0 && y_out < WIDTH-MASK_WIDTH+1)
    {
      float value = 0.0f;
      for (int i = 0; i < MASK_WIDTH; i++)
      {
          for (int j = 0; j < MASK_WIDTH; j++)
          {
              value += filter[i * MASK_WIDTH + j] * in_tile[][];
          }
      }
      out[y_out * OUT_WIDTH + x_out] = value;
    }

    




    // Compute output value for the valid region
    if (tx < TILE_SIZE - PAD && ty < TILE_SIZE - PAD)
    {
        if (x < n && y < n)
        {
            

            out[x * n + y] = value;
        }
    }



}

int main(int argc, char **argv) {
  size_t size_in = IN_WIDTH * IN_WIDTH * sizeof(float);
  size_t size_filter = MASK_WIDTH * MASK_WIDTH * sizeof(float);
  size_t size_out = OUT_WIDTH * OUT_WIDTH * sizeof(float);

  // Allocate and initialize host memory
  float *h_in = (float *)malloc(size_in);
  float *h_filter = (float *)malloc(size_filter);
  float *h_out = (float *)malloc(size_out);

  for (int i = 0; i < IN_WIDTH * IN_WIDTH; i++)
  {
    h_in[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
  {
    h_filter[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_in, *d_filter, *d_out;
  cudaMalloc((void **)&d_in, size_in);
  cudaMalloc((void **)&d_filter, size_filter);
  cudaMalloc((void **)&d_out, size_out);

  // Copy data from host to device
  cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

  

  // Create CUDA event to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start);


  // Launch the kernel
  int numBlock = (IN_WIDTH + TILE - 1) / tmp;
  dim3 blockDim(TILE, TILE);
  dim3 gridDim(numBlock, numBlock);
  kernel<<<gridDim, blockDim>>>(d_in, d_filter, d_out);

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
  for (int y_in = 0; i < IN_WIDTH; i++)
  {
    for (int x_in = 0; j < IN_WIDTH; j++)
    {
      int y_out = y_out - MASK_WIDTH + 1;
      int x_out = x_out - MASK_WIDTH + 1;

      if (y_in >= 0 && y_in < OUT_WIDTH && x_in >= 0 && x_in < OUT_WIDTH)
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
            gold[y_in*OUT_WIDTH+x_in] = value;
      }      
    }
  }



  for (int i = 0; i < OUT_WIDTH*OUT_WIDTH; i++)
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

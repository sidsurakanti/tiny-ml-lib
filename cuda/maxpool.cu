#include "errors.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void MaxPoolKernel() {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int row = tx + bx * blockDim.x;
  int col = ty + by * blockDim.y;
  int chan = bz * blockDim.z; // assume tz == 1

  printf("hello from thread (%d, %d) in channel %d\n", row, col, chan);
}

int main() {
  std::cout << "launching gpu kernel..." << std::endl;

  dim3 gridDim(1, 1, 4); // blocks in grid
  dim3 blockDim(2, 2);   // threads in block

  MaxPoolKernel<<<gridDim, blockDim>>>();
  CU_CHECK(cudaDeviceSynchronize());

  std::cout << "job finished!" << std::endl;

  return 0;
}

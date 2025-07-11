#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
  printf("hello from gpu thread#[%d, %d]\n", blockIdx.x, threadIdx.x);
}

int main() {
  std::cout << "launching gpu kernel..." << std::endl;

  hello<<<1, 4>>>();
  cudaDeviceSynchronize();

  std::cout << "job finished!" << std::endl;

  return 0;
}

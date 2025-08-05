#include <cuda_runtime.h>

__global__ void ReluKernel(float *X, float *C, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row >= m || col >= n)
    return;

  float &value = X[row * n + col];
  C[row * n + col] = value > 0 ? value : 0;
}

__global__ void ReluBackKernel(float *dZ, float *X, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);
  if (row < m && col < n) {
    int idx = row * n + col;
    X[idx] = dZ[idx] * (X[idx] > 0);
  }
}

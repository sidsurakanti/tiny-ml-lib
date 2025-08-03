#include "errors.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

void printMat(float *mat, int size, int colWidth) {
  for (int i = 0; i < size; i++) {
    printf("%6.2f ", mat[i]);
    if ((i + 1) % colWidth == 0)
      printf("\n");
  }
}

template <int kernel_size = 2>
__global__ void MaxPoolKernel(float *mat, float *out, int m, int n, int outN,
                              int stride = 2) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  // int bz = blockIdx.z;

  float buffer[kernel_size][kernel_size];

  // int chan = bz * blockDim.z; // assume tz == 1
  int x = tx + bx * blockDim.x;
  int y = ty + by * blockDim.y;

  int rowStart = x * stride;
  int colStart = y * stride;

  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      int row = rowStart + i;
      int col = colStart + j;
      buffer[i][j] = mat[row * n + col];
    }
  }

  float max = -CUDART_INF;
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      max = fmaxf(buffer[i][j], max);
    }
  }

  out[x * outN + y] = max;
}

void fillMat(float *mat, int size) {
  srand(3333);
  for (int i = 0; i < size; i++) {
    mat[i] = ((float)(rand() % 10));
  }
}

int main() {
  std::cout << "launching gpu kernel..." << std::endl;

  dim3 dimsM(4, 4);
  int size = dimsM.x * dimsM.y;
  int memsize = size * sizeof(float);
  float *mat_h;

  CU_CHECK(cudaMallocHost(&mat_h, memsize));

  dim3 dimsOut(2, 2);
  int sizeOut = dimsOut.x * dimsOut.y;
  int memsizeOut = sizeOut * sizeof(float);
  float *out_h;
  CU_CHECK(cudaMallocHost(&out_h, memsizeOut));

  fillMat(mat_h, size);
  fillMat(out_h, sizeOut);
  dim3 gridDim(1, 1, 4); // blocks in grid
  dim3 blockDim(2, 2);   // threads in block

  float *mat_d, *out_d;
  cudaMalloc(&mat_d, memsize);
  cudaMalloc(&out_d, memsizeOut);
  cudaMemcpy(mat_d, mat_h, memsize, cudaMemcpyHostToDevice);

  MaxPoolKernel<<<gridDim, blockDim>>>(mat_d, out_d, dimsM.x, dimsM.y,
                                       dimsOut.y);
  CU_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(out_h, out_d, memsizeOut, cudaMemcpyDeviceToHost);

  printMat(mat_h, size, dimsM.y);
  printf("\n");
  printMat(out_h, sizeOut, dimsOut.y);

  std::cout << "job finished!" << std::endl;
  return 0;
}

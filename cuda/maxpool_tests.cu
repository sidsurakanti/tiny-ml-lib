#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

#include "errors.cuh"
#include "maxpool.cuh"
#include "maxpool_utils.hpp"

int main() {
  std::cout << "launching gpu kernel..." << std::endl;

  // N, H, W (we're gonna just put in C in the kernel)
  nchw dimsIn{2000, 256, 28, 28};
  int sizeIn = dimsIn.n * dimsIn.c * dimsIn.h * dimsIn.w;
  int memsizeIn = sizeIn * sizeof(float);
  float *in_h;

  CU_CHECK(cudaMallocHost(&in_h, memsizeIn));
  int kernelSize = 2;
  int stride = 2;

  nchw dimsOut{dimsIn.n, dimsIn.c,
               calcOutDim(dimsIn.h, kernelSize, stride, true),
               calcOutDim(dimsIn.w, kernelSize, stride, true)};
  int sizeOut = dimsOut.n * dimsOut.c * dimsOut.h * dimsOut.w;
  int memsizeOut = sizeOut * sizeof(float);
  float *out_h;

  cudaEvent_t start, stop;
  float elapsed;
  CU_CHECK(cudaEventCreate(&start));
  CU_CHECK(cudaEventCreate(&stop));

  printf("[RUNNING] MaxPool\n");
  CU_CHECK(cudaEventRecord(start));

  CU_CHECK(cudaMallocHost(&out_h, memsizeOut));

  fillMat(in_h, sizeIn);
  fillMat(out_h, sizeOut);

  int BLOCK_SIZE = 16;
  int gridRows = (dimsOut.h + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int gridCols = (dimsOut.w + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // n, ceil(out_h) / block_size, same for w
  dim3 gridDim(gridCols, gridRows, dimsOut.n); // blocks in grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);       // threads in block

  float *in_d, *out_d;
  cudaMalloc(&in_d, memsizeIn);
  cudaMalloc(&out_d, memsizeOut);
  cudaMemcpy(in_d, in_h, memsizeIn, cudaMemcpyHostToDevice);

  printf("in shape (%d, %d, %d, %d)\n", dimsIn.n, dimsIn.c, dimsIn.h, dimsIn.w);
  printf("out shape (%d, %d, %d, %d)\n", dimsOut.n, dimsOut.c, dimsOut.h,
         dimsOut.w);

  MaxPoolKernel<<<gridDim, blockDim>>>(in_d, out_d, dimsIn.n, dimsIn.c,
                                       dimsIn.h, dimsIn.w, dimsOut.h, dimsOut.w,
                                       kernelSize, stride);

  CU_CHECK(cudaDeviceSynchronize());

  CU_CHECK(cudaEventRecord(stop));
  CU_CHECK(cudaEventSynchronize(stop));
  CU_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  printf("[TIME] Completed in %.2fms.\n", elapsed);

  CU_CHECK(cudaEventDestroy(start));
  CU_CHECK(cudaEventDestroy(stop));

  cudaMemcpy(out_h, out_d, memsizeOut, cudaMemcpyDeviceToHost);

  // printMat(in_h, dimsIn);
  // printf("\n");
  // printMat(out_h, dimsOut);

  std::cout << "job finished!" << std::endl;
  return 0;
}

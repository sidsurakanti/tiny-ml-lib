#pragma once
#include <cuda_runtime.h>

template <int block_size>
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n, int k);

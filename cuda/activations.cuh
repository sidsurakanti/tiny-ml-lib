#pragma once
#include <cuda_runtime.h>

__global__ void ReluKernel(float *X, float *C, int m, int n);
__global__ void ReluBackKernel(float *dZ, float *X, int m, int n);

#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>

template <int block_size>
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n, int k);

__global__ void ReluKernel(float *X, float *C, int m, int n);

template <int block_size>
__global__ void MatTransposeKernel(float *mat, float *buf, int m, int n);

__global__ void MatSumKernel(float *mat, float *dst, int m, int n);

__global__ void MatMatSubKernel(float *mat, float *subber, float c, int m,
                                int n);

__global__ void VecMatAddKernel(float *vec, float *mat, int m, int n);
void vecMatAdd(float *vec, float *mat, int m, int n);

__global__ void VecMatDivKernel(int c, float *mat, int m, int n);
void vecMatDiv(int c, float *mat, int m, int n);

__global__ void ReluBackKernel(float *dZ, float *X, int m, int n);

#endif

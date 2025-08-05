#pragma once
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

float *matTranspose(float *mat, int m, int n);
void matMul(float *A, float *B, float *C, int m, int n, int k);
void matSum(float *mat, float *dst, int m, int n);
void matMatSub(py::capsule mat, py::capsule subber, float c, int m, int n);
void vecMatAdd(float *vec, float *mat, int m, int n);
void vecMatDiv(int c, float *mat, int m, int n);

template <int block_size>
__global__ void MatTransposeKernel(float *mat, float *buf, int m, int n);

__global__ void VecMatAddKernel(float *vec, float *mat, int m, int n);
__global__ void VecMatDivKernel(int c, float *mat, int m, int n);
__global__ void MatSumKernel(float *mat, float *dst, int m, int n);
__global__ void MatMatSubKernel(float *mat, float *subber, float c, int m,
                                int n);

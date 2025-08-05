#include "errors.cuh"
#include "mat_ops.cuh"
#include "matmul.cuh"
#include <cstdio>

#include <cstdlib>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

const int BLOCK_SIZE = 16;

void vecMatDiv(int c, float *mat, int m, int n) {
  int block_size = 32;
  dim3 blockDim(block_size, block_size);
  unsigned int gridRows = (m + block_size - 1) / block_size;
  unsigned int gridCols = (n + block_size - 1) / block_size;
  dim3 gridDim(gridCols, gridRows);

  VecMatDivKernel<<<gridDim, blockDim>>>(c, mat, m, n);
  CU_CHECK(cudaDeviceSynchronize());
}

__global__ void VecMatAddKernel(float *vec, float *mat, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row >= m || col >= n)
    return;

  mat[row * n + col] += vec[col];
}

float *matTranspose(float *mat, int m, int n) {
  // create output buff
  float *buf;
  CU_CHECK(cudaMalloc(&buf, m * n * sizeof(float)));

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block
  MatTransposeKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(mat, buf, m, n);
  CU_CHECK(cudaDeviceSynchronize());
  CU_CHECK(cudaGetLastError());
  return buf;
}

void matMul(float *A, float *B, float *C, int m, int n, int k) {
  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows);      // blocks per grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  MatMulKernel<BLOCK_SIZE><<<gridSize, blockSize>>>(A, B, C, m, n, k);
  CU_CHECK(cudaGetLastError());
}

// equiv to np.sum(mat, axis=0)
void matSum(float *mat, float *dst, int m, int n) {
  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows);      // blocks per grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block
  MatSumKernel<<<gridSize, blockSize>>>(mat, dst, m, n);
  CU_CHECK(cudaGetLastError());
}

// mat A - B
void matMatSub(py::capsule mat, py::capsule subber, float c, int m, int n) {
  float *ptrMat = static_cast<float *>(mat.get_pointer()); // batchsize * inputs
  float *ptrSubber =
      static_cast<float *>(subber.get_pointer()); // inputs * outputs

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows);      // blocks per grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  MatMatSubKernel<<<gridSize, blockSize>>>(ptrMat, ptrSubber, c, m, n);
  CU_CHECK(cudaGetLastError());
}

__global__ void MatMatSubKernel(float *mat, float *subber, float c, int m,
                                int n);
// explicit declarations
template __global__ void MatTransposeKernel<16>(float *mat, float *buf, int m,
                                                int n);

template <int block_size>
__global__ void MatTransposeKernel(float *mat, float *buf, int m, int n) {
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = ty + (blockIdx.y * blockDim.y);
  int col = tx + (blockIdx.x * blockDim.x);

  __shared__ float tile[block_size][block_size];

  // load element into shared mem
  if (row < m && col < n)
    tile[ty][tx] = mat[row * n + col];
  else
    tile[ty][tx] = 0.0f;

  __syncthreads();

  if (row < m && col < n)
    buf[col * m + row] = tile[ty][tx];
}

__global__ void MatSumKernel(float *mat, float *dst, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row < m && col < n)
    atomicAdd(&dst[col], mat[row * n + col]);
}

__global__ void MatMatSubKernel(float *mat, float *subber, float c, int m,
                                int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row < m && col < n)
    atomicAdd(&mat[row * n + col], -(c * subber[row * n + col]));
}

__global__ void VecMatDivKernel(int c, float *mat, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row >= m || col >= n)
    return;

  mat[row * n + col] /= c;
}

// make sure vec and mat are 1d row majored otherwise you're cooked bro
void vecMatAdd(float *vec, float *mat, int m, int n) {
  int block_size = 32;
  dim3 blockDim(block_size, block_size);
  unsigned int gridRows = (m + block_size - 1) / block_size;
  unsigned int gridCols = (n + block_size - 1) / block_size;
  dim3 gridDim(gridCols, gridRows);

  VecMatAddKernel<<<gridDim, blockDim>>>(vec, mat, m, n);
  CU_CHECK(cudaDeviceSynchronize());
}

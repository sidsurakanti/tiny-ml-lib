#include "errors.cuh"
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

const int BLOCK_SIZE = 16;

typedef typename py::array_t<float, py::array::c_style | py::array::forcecast>
    py_ndarray_t;

py::array matMul(py_ndarray_t A, py_ndarray_t B, int m, int n, int k) {
  unsigned int size_A = m * n;
  unsigned int size_B = n * k;
  unsigned int size_C = m * k;

  unsigned int mem_sizeA = sizeof(float) * size_A;
  unsigned int mem_sizeB = sizeof(float) * size_B;
  unsigned int mem_sizeC = sizeof(float) * size_C;

  // unchecked is better than arr.request() for buf & then buf.ptr()
  // cus it auto throws when given more than 1d
  const float *A_h = A.unchecked<1>().data(0); // ptr to A[0]
  const float *B_h = B.unchecked<1>().data(0);
  float *C_h = (float *)calloc(m * k, sizeof(float));

  float *A_d, *B_d, *C_d;

  CU_CHECK(cudaMalloc(&A_d, mem_sizeA));
  CU_CHECK(cudaMalloc(&B_d, mem_sizeB));
  CU_CHECK(cudaMalloc(&C_d, mem_sizeC));

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  CU_CHECK(
      cudaMemcpyAsync(A_d, A_h, mem_sizeA, cudaMemcpyHostToDevice, stream));
  CU_CHECK(
      cudaMemcpyAsync(B_d, B_h, mem_sizeB, cudaMemcpyHostToDevice, stream));

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows);      // blocks per grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  MatMulKernel<BLOCK_SIZE>
      <<<gridSize, blockSize, 1, stream>>>(A_d, B_d, C_d, m, n, k);

  // wait host thread & error check
  CU_CHECK(cudaGetLastError());
  CU_CHECK(cudaStreamSynchronize(stream));

  // copy result back to host
  CU_CHECK(
      cudaMemcpyAsync(C_h, C_d, mem_sizeC, cudaMemcpyDeviceToHost, stream));
  CU_CHECK(cudaStreamSynchronize(stream));

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaStreamDestroy(stream);

  // wrap the c ptr in a capsule with a destructor so we can safely pass it
  // around we need to do this so when python destruct's its capsule obj we auto
  // free the memory for the c ptr
  py::capsule free_when_done(C_h, [](void *ptr) { free(ptr); });
  py::array_t<float> result = py::array_t<float>({m, k}, // shape
                                                 C_h,    // data ptr
                                                 free_when_done);

  return result;
}

void init_matmul(py::module_ &m) {
  m.def("matmul", &matMul,
        "Matrix multiplication: A @ B = C\n"
        "Args:\n"
        "  A: 1D array, shape (m*n) representing (m, n) matrix in row-major "
        "order\n"
        "  B: 1D array, shape (n*k) representing (n, k) matrix in row-major "
        "order\n"
        "  m, n, k: matrix dimensions\n"
        "Returns:\n"
        "  C: 2D array, shape (m, k)");
}

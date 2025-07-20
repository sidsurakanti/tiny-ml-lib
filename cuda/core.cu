#include "errors.cuh"
#include "matmul.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

const int BLOCK_SIZE = 16;

py::capsule makeCapsule(void *ptr, bool isCudaPtr) {
  auto deleter =
      isCudaPtr ? [](void *p) { cudaFree(p); } : [](void *p) { free(p); };

  return py::capsule(ptr, deleter);
}

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
  py::capsule free_when_done = makeCapsule(C_h, false);
  py::array_t<float> result = py::array_t<float>({m, k}, // shape
                                                 C_h,    // data ptr
                                                 free_when_done);

  return result;
}

py::capsule toGPU(py_ndarray_t obj, int size) {
  int memsize = size * sizeof(float);
  float *ptr;

  CU_CHECK(cudaMalloc(&ptr, memsize));
  CU_CHECK(cudaMemset(ptr, 0, memsize));

  return makeCapsule(ptr, true);
}

void updateGpuMemory(py::capsule cap, int size) {
  float *ptrH = static_cast<float *>(cap.get_pointer());
  float *ptrD;

  CU_CHECK(
      cudaMemcpy(ptrD, ptrH, size * sizeof(float), cudaMemcpyHostToDevice));
}

auto initBuffers(py_ndarray_t W, int input_size, int output_size,
                 int batch_size) {

  const float *W_h = W.unchecked<1>().data(0); // ptr to weights

  // init W, b, dW, dB, C
  int sizeW = input_size * output_size;
  int sizeB = 1 * output_size;
  int sizeC = batch_size * output_size;

  int memsizeW = sizeW * sizeof(float);
  int memsizeB = sizeB * sizeof(float);
  int memsizeC = sizeC * sizeof(float);
  float *ptrW, *ptrB, *ptrC, *ptrdW, *ptrdB;

  CU_CHECK(cudaMalloc(&ptrW, memsizeW));
  CU_CHECK(cudaMalloc(&ptrB, memsizeB));
  CU_CHECK(cudaMalloc(&ptrC, memsizeC));
  CU_CHECK(cudaMalloc(&ptrdW, memsizeW));
  CU_CHECK(cudaMalloc(&ptrdB, memsizeB));

  CU_CHECK(cudaMemset(ptrB, 0, memsizeC));
  CU_CHECK(cudaMemset(ptrC, 0, memsizeC));
  CU_CHECK(cudaMemset(ptrW, 0, memsizeW));
  CU_CHECK(cudaMemset(ptrB, 0, memsizeB));

  // copy over weight inits
  CU_CHECK(cudaMemcpy(ptrW, W_h, memsizeW, cudaMemcpyHostToDevice));

  py::capsule w = makeCapsule(ptrW, true);
  py::capsule b = makeCapsule(ptrB, true);
  py::capsule c = makeCapsule(ptrC, true);
  py::capsule dw = makeCapsule(ptrdW, true);
  py::capsule db = makeCapsule(ptrdB, true);

  // access with capsule.get_pointer()
  return std::make_tuple(w, b, c, dw, db);
}

void init_matmul(py::module_ &m) {
  m.def("toGPU", &toGPU);
  m.def("updateGpuMemory", &updateGpuMemory);
  m.def("initBuffers", &initBuffers);
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

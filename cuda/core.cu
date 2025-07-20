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

void relu(py::capsule mat, int m, int n) {
  float *ptr = static_cast<float *>(mat.get_pointer());

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  ReluKernel<<<gridDim, blockDim>>>(ptr, m, n);
}

// we're gonna pass these capsules in from python and let them handle how they
// use the capsules we've init'd
// assume they're already on host and we just have to perform op
void linear(py::capsule X, py::capsule W, py::capsule b, py::capsule Y,
            int inputs, int outputs, int batch_size) {
  float *ptrX = static_cast<float *>(X.get_pointer()); // batchsize * inputs
  float *ptrW = static_cast<float *>(W.get_pointer()); // inputs * outputs
  float *ptrB = static_cast<float *>(b.get_pointer()); // 1 * outputs
  float *ptrY = static_cast<float *>(Y.get_pointer()); // batchsize * outputs

  int &m = batch_size;
  int &n = inputs;
  int &k = outputs;

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  MatMulKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(ptrX, ptrW, ptrY, m, n, k);

  // wait host thread & error check
  CU_CHECK(cudaGetLastError());      // launch errors
  CU_CHECK(cudaDeviceSynchronize()); // kernel errors

  // output matrix is of size (batchsize, outputs) so we need a kernel size of
  // that
  unsigned int &vGridRows = gridRows;
  unsigned int vGridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 vGridDim(vGridCols, vGridRows);
  VecMatAddKernel<<<vGridDim, blockDim>>>(ptrB, ptrY, m, n);
}

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

py::capsule initBuff(int m, int n) {
  int memsize = (m * n) * sizeof(float);
  float *ptr;

  CU_CHECK(cudaMalloc(&ptr, memsize));
  CU_CHECK(cudaMemset(ptr, 0, memsize));

  return makeCapsule(ptr, true);
}

py::capsule toGPU(py_ndarray_t obj, int size) {
  const float *dataPtr = obj.unchecked<1>().data(0); // ptr to weights
  int memsize = size * sizeof(float);
  float *ptr;

  CU_CHECK(cudaMalloc(&ptr, memsize));
  CU_CHECK(cudaMemcpy(ptr, dataPtr, memsize, cudaMemcpyHostToDevice));

  return makeCapsule(ptr, true);
}

py::array toCPU(py::capsule cap, int m, int n) {
  float *ptr = static_cast<float *>(cap.get_pointer());
  int memsize = m * n * sizeof(float);
  float *retPtr = (float *)malloc(memsize);

  CU_CHECK(cudaMemcpy(retPtr, ptr, memsize, cudaMemcpyDeviceToHost));
  py::array_t<float> result = py::array_t<float>({m, n}, // shape
                                                 retPtr, // data ptr
                                                 makeCapsule(retPtr, false));
  return result;
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

  // copy over weight inits
  CU_CHECK(cudaMemcpy(ptrW, W_h, memsizeW, cudaMemcpyHostToDevice));

  // set all new init'd mat's to 0
  CU_CHECK(cudaMemset(ptrdW, 0, memsizeW));
  CU_CHECK(cudaMemset(ptrB, 0, memsizeB));
  CU_CHECK(cudaMemset(ptrdB, 0, memsizeB));
  CU_CHECK(cudaMemset(ptrC, 0, memsizeC));

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
  m.def("toCPU", &toCPU);
  m.def("updateGpuMemory", &updateGpuMemory);
  m.def("initBuff", &initBuff);
  m.def("initBuffers", &initBuffers);
  m.def("linear", &linear);
  m.def("relu", &relu);
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

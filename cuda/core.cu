#include "errors.cuh"
#include "matmul.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

const int BLOCK_SIZE = 16;
typedef typename py::array_t<float, py::array::c_style | py::array::forcecast>
    py_ndarray_t;

py::capsule makeCapsule(void *ptr, bool isCudaPtr) {
  auto deleter =
      isCudaPtr ? [](void *p) { cudaFree(p); } : [](void *p) { free(p); };

  return py::capsule(ptr, deleter);
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

// we're gonna pass these capsules in from python and let them handle how they
// use the capsules we've init'd
// assume they're already on device and we just have to perform op
void linear(py::capsule X, py::capsule W, py::capsule b, py::capsule C,
            int batch_size, int inputs, int outputs) {
  float *ptrX = static_cast<float *>(X.get_pointer()); // batchsize * inputs
  float *ptrW = static_cast<float *>(W.get_pointer()); // inputs * outputs
  float *ptrB = static_cast<float *>(b.get_pointer()); // 1 * outputs
  float *ptrC = static_cast<float *>(C.get_pointer()); // batchsize * outputs

  int &m = batch_size;
  int &n = inputs;
  int &k = outputs;

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  MatMulKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(ptrX, ptrW, ptrC, m, n, k);

  // dont have to cuda dev sync here because everything in the same thread runs
  // sequentially
  CU_CHECK(cudaGetLastError()); // launch errors

  vecMatAdd(ptrB, ptrC, m, k);
}

// write result (dX) to input from L-1 (self.X)
void linearBack(py::capsule X, py::capsule W, py::capsule dW, py::capsule dB,
                py::capsule dZ, int batch_size, int inputs, int outputs) {
  float *ptrX =
      static_cast<float *>(X.get_pointer()); // batchsize * inputs (m, n)
  float *ptrW =
      static_cast<float *>(W.get_pointer()); // inputs * outputs (n, k)
  float *ptrdW =
      static_cast<float *>(dW.get_pointer()); // inputs * outputs (n, k)
  float *ptrdB = static_cast<float *>(dB.get_pointer()); // 1 * outputs (1, k)
  float *ptrdZ =
      static_cast<float *>(dZ.get_pointer()); // batchsize * outputs (m, k)

  int &m = batch_size;
  int &n = inputs;
  int &k = outputs;

  // zero grads before refilling
  // not needed b/c we reassign values but im just putting this here
  cudaMemset(ptrdW, 0, n * k * sizeof(float));
  cudaMemset(ptrdB, 0, k * sizeof(float));

  // calculate dW
  // (m, n).T * (m, k) = (n, k)
  // x.T @ dZ / m = dW
  float *xT = matTranspose(ptrX, m, n);
  matMul(xT, ptrdZ, ptrdW, n, m, k);
  vecMatDiv(m, ptrdW, n, k);
  cudaFree(xT);

  // calc dB
  // sum(dZ) / m
  // sum(m, k) = (1, k)
  // zero out prev dB
  cudaMemset(ptrdB, 0, sizeof(float) * outputs);
  matSum(ptrdZ, ptrdB, m, k);
  vecMatDiv(m, ptrdB, 1, k);

  // calc dX (override ptrX bc dZ is just gonna be the layers output buff)
  // (m, k) * (n, k).T = (m, n)
  // dZ @ w.T
  float *wT = matTranspose(ptrW, n, k);
  matMul(ptrdZ, wT, ptrX, m, k, n);
  cudaFree(wT);
}

void relu(py::capsule X, py::capsule C, int m, int n) {
  float *ptrX = static_cast<float *>(X.get_pointer());
  float *ptrC = static_cast<float *>(C.get_pointer());

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  ReluKernel<<<gridDim, blockDim>>>(ptrX, ptrC, m, n);
  CU_CHECK(cudaGetLastError());
}

void reluBack(py::capsule dZ, py::capsule X, int m, int n) {
  float *ptrX =
      static_cast<float *>(X.get_pointer()); // batchsize * inputs (m, n)
  float *ptrdZ =
      static_cast<float *>(dZ.get_pointer()); // batchsize * outputs (m, k)

  unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(gridCols, gridRows);      // blocks per grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  // write res to ptrX
  ReluBackKernel<<<gridDim, blockDim>>>(ptrdZ, ptrX, m, n);
  CU_CHECK(cudaGetLastError());
}

py::array npMatMul(py_ndarray_t A, py_ndarray_t B, int m, int n, int k) {
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

void updateGpuMemory(py_ndarray_t arr, py::capsule gpuPtr, int m, int n) {
  float *ptrD = static_cast<float *>(gpuPtr.get_pointer());
  const float *ptrH = arr.unchecked<1>().data(0);

  CU_CHECK(
      cudaMemcpy(ptrD, ptrH, m * n * sizeof(float), cudaMemcpyHostToDevice));
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
  m.def("linearBack", &linearBack);
  m.def("matMatSub", &matMatSub);
  m.def("relu", &relu);
  m.def("reluBack", &reluBack);
  m.def("matmul", &npMatMul,
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

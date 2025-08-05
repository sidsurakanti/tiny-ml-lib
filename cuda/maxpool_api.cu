#include "core.cu"
#include "errors.cuh"
#include "maxpool.cu"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
typedef typename py::array_t<float, py::array::c_style | py::array::forcecast>
    py_ndarray_t;

py::array maxPoolCpuRes(py_ndarray_t input, int n, int w, int c, int h,
                        int kernel_size = 2, int stride = 2,
                        bool use_padding = true) {
  const float *in_h = input.unchecked<1>().data(0); // ptr to A[0]
  nchw dimsIn{n, c, h, w};
  int sizeIn = dimsIn.n * dimsIn.c * dimsIn.h * dimsIn.w;
  int memsizeIn = sizeIn * sizeof(float);

  nchw dimsOut{dimsIn.n, dimsIn.c,
               calcOutDim(dimsIn.h, kernel_size, stride, use_padding),
               calcOutDim(dimsIn.w, kernel_size, stride, use_padding)};

  int sizeOut = dimsOut.n * dimsOut.c * dimsOut.h * dimsOut.w;
  int memsizeOut = sizeOut * sizeof(float);
  float *out_h;

  CU_CHECK(cudaMallocHost(&out_h, memsizeOut));

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

  MaxPoolKernel<<<gridDim, blockDim>>>(in_d, out_d, dimsIn.n, dimsIn.c,
                                       dimsIn.h, dimsIn.w, dimsOut.h, dimsOut.w,
                                       kernel_size, stride);

  cudaMemcpy(out_h, out_d, memsizeOut, cudaMemcpyDeviceToHost);

  py::capsule free_when_done = makeCapsule(out_h, false);
  py::array_t<float> result =
      py::array_t<float>({dimsOut.n, dimsOut.c, dimsOut.h, dimsOut.w}, // shape
                         out_h, // data ptr
                         free_when_done);

  return result;
}

void init_maxpool(py::module &m) { m.def("maxpool", &maxPoolCpuRes); }

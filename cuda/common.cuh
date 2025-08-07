#pragma once
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

typedef typename py::array_t<float, py::array::c_style | py::array::forcecast>
    py_ndarray_t;

inline py::capsule makeCapsule(void *ptr, bool isCudaPtr,
                               bool isPinnedHost = false) {
  auto deleter = isCudaPtr      ? [](void *p) { cudaFree(p); }
                 : isPinnedHost ? [](void *p) { cudaFreeHost(p); }
                                : [](void *p) { free(p); };

  return py::capsule(ptr, deleter);
}

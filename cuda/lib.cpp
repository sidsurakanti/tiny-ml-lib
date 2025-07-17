#include <pybind11/pybind11.h>

namespace py = pybind11;

// declarations for inits of other files
void init_matmul(py::module_ &);
void init_hello(py::module_ &);

PYBIND11_MODULE(lib, m) {
  init_matmul(m);
  init_hello(m);
  m.doc() = "core";
}

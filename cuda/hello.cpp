#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::string hello(std::string name) {
  return "Hello, " + name + ".";
}

void init_hello(py::module_ &m) {
  m.def("hello", &hello);
}

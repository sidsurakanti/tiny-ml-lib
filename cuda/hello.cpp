#include <string>
#include <pybind11/pybind11.h>

std::string hello(std::string name, int n) {
  return "Hello, " + name + ". [#" + std::to_string(n) + "]";
}

PYBIND11_MODULE(lib, handle) {
  handle.doc() = "1234";
  handle.def("hello", &hello);
}

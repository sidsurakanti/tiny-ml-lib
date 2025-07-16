cmake_minimum_required(VERSION 3.10)
project(lib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

# for lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_FINDPYTHON NEW)

add_subdirectory(extern/pybind11)

pybind11_add_module(lib
  # cuda/matmul.cu 
  cuda/hello.cpp
)

# set cmake install dir to root of proj so we don't have it installed in local
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "install prefix" FORCE)

install(TARGETS lib DESTINATION native)

#pragma once

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("[CUDA ERROR] in %s:%d\n", __FILE__, __LINE__);                   \
      printf("[ERR CODE] %d\n", error_code);                                   \
      printf("[ERR] %s\n", cudaGetErrorString(error_code));                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

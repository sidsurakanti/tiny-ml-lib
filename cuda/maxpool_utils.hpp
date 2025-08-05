#pragma once
#include <cstdio>
#include <cstdlib>

struct nchw {
  int n, c, w, h;
};

inline void printTensor(float *mat, nchw size) {
  auto [n, c, w, h] = size;
  printf("\n");
  for (int i = 0; i < (n * c * w * h); i++) {
    printf("%6.2f ", mat[i]);
    if ((i + 1) % (w * h) == 0)
      printf("\n");
    if ((i + 1) % w == 0)
      printf("\n");
    else if ((i + 1) % (c * w * h) == 0)
      printf("\n");
  }
}

inline void fillMat(float *mat, int size) {
  srand(3333);
  for (int i = 0; i < size; i++) {
    mat[i] = ((float)(rand() % 10));
  }
}

inline int calcOutDim(int input_size, int kernel_size, int stride, bool use_padding) {
  if (use_padding)
    return (input_size + stride - 1) / stride; // ceil(input / stride)
  else
    return (input_size - kernel_size) / stride + 1;
}

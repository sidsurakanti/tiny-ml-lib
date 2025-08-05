#include "maxpool.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <math_constants.h>

// N, C, H, W
template <int max_kernel_size> // workaround
__global__ void MaxPoolKernel(float *in, float *out, int N, int C, int H, int W,
                              int outH, int outW, int kernel_size, int stride) {
  // assume we use only 2d blocks
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 3d grid (N, H, W)
  int bz = blockIdx.z; // n (batch i)
  int by = blockIdx.y;
  int bx = blockIdx.x;

  // works for 1 channel
  float buffer[max_kernel_size][max_kernel_size];

  int batch = bz * blockDim.z; // assume tz == 0 (n)
  // row, col for out matrix
  int outRow = ty + by * blockDim.y;
  int outCol = tx + bx * blockDim.x;

  if (outRow >= outH || outCol >= outW)
    return;

  int rowStart = outRow * stride;
  int colStart = outCol * stride;

  // processing each channel is very small amount of work for each thread
  // so we can make it work on all the channels (reasonably < 512)
  for (int c = 0; c < C; c++) {
    // load kernel into buffer
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        int row = rowStart + i;
        int col = colStart + j;
        int idx = ((batch * C * H * W) + (c * H * W)) + row * W + col;

        if (row >= H || col >= W) // padding for edge cases
          buffer[i][j] = 0.0f;
        else
          buffer[i][j] = in[idx];
      }
    }

    // apply maxpool on the buffer
    float max = -CUDART_INF;
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        max = fmaxf(buffer[i][j], max);
      }
    }

    // printf("%f", max);

    // add res to output buf
    out[((batch * C * outH * outW) + (c * outH * outW)) + (outRow * outW) +
        outCol] = max;
  }
}

template __global__ void MaxPoolKernel<15>(float *in, float *out, int N, int C,
                                           int H, int W, int outH, int outW,
                                           int kernel_size, int stride);

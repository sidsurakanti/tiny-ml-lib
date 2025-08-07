#include "maxpool.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <math_constants.h>

// N, C, H, W
template <int max_kernel_size> // workaround
__global__ void MaxPoolKernel(float *in, float *out, int N, int C, int H, int W,
                              int outH, int outW, int kernel_size, int stride,
                              int *backwards_mask) {
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
  int batchIdx = (batch * C * H * W);

  // processing each channel is very small amount of work for each thread
  // so we can make it work on all the channels (reasonably < 512)
  for (int c = 0; c < C; c++) {
    int chanIdx = batchIdx + (c * H * W);

    // load kernel on buff
    for (int i = 0; i < kernel_size; i++) {
      int row = rowStart + i;
      int rowIdx = rowStart * W;

      for (int j = 0; j < kernel_size; j++) {
        int col = colStart + j;

        if (row >= H || col >= W) // padding for edge cases
          buffer[i][j] = 0.0f;
        else
          buffer[i][j] = in[chanIdx + rowIdx + col];
      }
    }

    // apply maxpool on the buffer
    float max = -CUDART_INF;
    int max_idx[2] = {};
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++) {
        if (buffer[i][j] > max) {
          max = buffer[i][j];
          max_idx[0] = i;
          max_idx[1] = j;
        }
      }
    }

    // set backwards mask so we can use it for backprop
    int out_idx = ((batch * C * outH * outW) + (c * outH * outW)) +
                  (outRow * outW) + outCol;

    // add res to output buf
    out[out_idx] = max;
    backwards_mask[out_idx] =
        chanIdx + (rowStart + max_idx[0]) * W + (colStart + max_idx[1]);

    // printf("%f", max);
  }
}

template __global__ void MaxPoolKernel<15>(float *in, float *out, int N, int C,
                                           int H, int W, int outH, int outW,
                                           int kernel_size, int stride,
                                           int *backwards_mask);

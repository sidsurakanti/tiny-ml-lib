#pragma once

template <int max_kernel_size = 15>
__global__ void MaxPoolKernel(float *in, float *out, int N, int C, int H, int W,
                              int outH, int outW, int kernel_size, int stride);

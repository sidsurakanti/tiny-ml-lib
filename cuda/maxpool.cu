#include "errors.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

struct nchw {
  int n, c, w, h;
};

int calcOutDim(int input_size, int kernel_size, int stride, bool use_padding) {
  if (use_padding)
    return (input_size + stride - 1) / stride; // ceil(input / stride)
  else
    return (input_size - kernel_size) / stride + 1;
}

void printMat(float *mat, nchw size) {
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

void fillMat(float *mat, int size) {
  srand(3333);
  for (int i = 0; i < size; i++) {
    mat[i] = ((float)(rand() % 10));
  }
}

template <int max_kernel_size = 15> // workaround
// N, C, H, W
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

int main() {
  std::cout << "launching gpu kernel..." << std::endl;

  // N, H, W (we're gonna just put in C in the kernel)
  nchw dimsIn{2000, 256, 28, 28};
  int sizeIn = dimsIn.n * dimsIn.c * dimsIn.h * dimsIn.w;
  int memsizeIn = sizeIn * sizeof(float);
  float *in_h;

  CU_CHECK(cudaMallocHost(&in_h, memsizeIn));
  int kernelSize = 2;
  int stride = 2;

  nchw dimsOut{dimsIn.n, dimsIn.c,
               calcOutDim(dimsIn.h, kernelSize, stride, true),
               calcOutDim(dimsIn.w, kernelSize, stride, true)};
  int sizeOut = dimsOut.n * dimsOut.c * dimsOut.h * dimsOut.w;
  int memsizeOut = sizeOut * sizeof(float);
  float *out_h;

  cudaEvent_t start, stop;
  float elapsed;
  CU_CHECK(cudaEventCreate(&start));
  CU_CHECK(cudaEventCreate(&stop));

  printf("[RUNNING] MaxPool\n");
  CU_CHECK(cudaEventRecord(start));

  CU_CHECK(cudaMallocHost(&out_h, memsizeOut));

  fillMat(in_h, sizeIn);
  fillMat(out_h, sizeOut);

  int BLOCK_SIZE = 16;
  int gridRows = (dimsOut.h + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int gridCols = (dimsOut.w + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // n, ceil(out_h) / block_size, same for w
  dim3 gridDim(gridCols, gridRows, dimsOut.n); // blocks in grid
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);       // threads in block

  float *in_d, *out_d;
  cudaMalloc(&in_d, memsizeIn);
  cudaMalloc(&out_d, memsizeOut);
  cudaMemcpy(in_d, in_h, memsizeIn, cudaMemcpyHostToDevice);

  printf("in shape (%d, %d, %d, %d)\n", dimsIn.n, dimsIn.c, dimsIn.h, dimsIn.w);
  printf("out shape (%d, %d, %d, %d)\n", dimsOut.n, dimsOut.c, dimsOut.h,
         dimsOut.w);

  MaxPoolKernel<<<gridDim, blockDim>>>(in_d, out_d, dimsIn.n, dimsIn.c,
                                       dimsIn.h, dimsIn.w, dimsOut.h, dimsOut.w,
                                       kernelSize, stride);

  CU_CHECK(cudaDeviceSynchronize());

  CU_CHECK(cudaEventRecord(stop));
  CU_CHECK(cudaEventSynchronize(stop));
  CU_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  printf("[TIME] Completed in %.2fms.\n", elapsed);

  CU_CHECK(cudaEventDestroy(start));
  CU_CHECK(cudaEventDestroy(stop));

  cudaMemcpy(out_h, out_d, memsizeOut, cudaMemcpyDeviceToHost);

  // printMat(in_h, dimsIn);
  // printf("\n");
  // printMat(out_h, dimsOut);

  std::cout << "job finished!" << std::endl;
  return 0;
}

#include "errors.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// ** lowk this file has a lot of dumb comments but that's js me thinking
// through the code
// ** feel free to ignore it

// BIG IDEA:
// set up matrices
// cuda malloc on device
// copy matrices from host -> device
// do matmul
// cpy back to host

#define BLOCK_SIZE 16

void matInit(float *mat, int size, int n) {
  for (int i = 0; i < size; i++) {
    mat[i] = n;
  }
}

// TODO: use shared memory to make this faster
template <int block_size>
// (m, n) * (n, k) = (m, k)
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n,
                             int k) {
  // NOTE: y for rows (vertical) in cuda and x (horiz) for cols
  int offsetY = blockDim.y * blockIdx.y;
  int offsetX = blockDim.x * blockIdx.x;

  // NOTE:
  // >>> think each thread == 1 element of C that we need to calculate
  // >>> so, using threadIdx.x and .y we can identify the row and col to
  // multiply for that specific element
  // >>> considering there's gonna be multiple
  // blocks per grid, we'd have to add the offset to get the actual global
  // threadIdx
  int row = threadIdx.y + offsetY;
  int col = threadIdx.x + offsetX;

  // bounds checking so we don't access uninit'd memory
  if (row >= m || col >= k)
    return;

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    // NOTE: have to index by row major bc everything is 1D
    sum += A[row * n + i] * B[k * i + col];
  }

  C[row * k + col] = sum;
  // printf("%f row: %d, col: %d, %f, %f\n ", cSum, row, col, A[wA * row + 1],
  //        B[wB * 1 + col]);
}

void matMul() {
  // set up data
  dim3 dimsA(60000, 784);
  dim3 dimsB(784, 258);
  dim3 dimsC(dimsA.x, dimsB.y);

  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int size_C = dimsC.x * dimsC.y;

  unsigned int mem_sizeA = sizeof(float) * size_A;
  unsigned int mem_sizeB = sizeof(float) * size_B;
  unsigned int mem_sizeC = sizeof(float) * size_C;

  float *A_h, *A_d, *B_h, *B_d, *C_h, *C_d;

  // alloc pinned memory on host for faster cpy times
  CU_CHECK(cudaMallocHost(&A_h, mem_sizeA));
  CU_CHECK(cudaMallocHost(&B_h, mem_sizeB));
  CU_CHECK(cudaMallocHost(&C_h, mem_sizeC));

  matInit(A_h, size_A, 1);
  matInit(B_h, size_B, 2);

  // allocate device mem
  CU_CHECK(cudaMalloc(&A_d, mem_sizeA));
  CU_CHECK(cudaMalloc(&B_d, mem_sizeB));
  CU_CHECK(cudaMalloc(&C_d, mem_sizeC));

  // copy matx's from host to device async
  // NOTE: seperate from host thread, so basically making this nonblocking
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  CU_CHECK(
      cudaMemcpyAsync(A_d, A_h, mem_sizeA, cudaMemcpyHostToDevice, stream));
  CU_CHECK(
      cudaMemcpyAsync(B_d, B_h, mem_sizeB, cudaMemcpyHostToDevice, stream));

  printf("[OPERATION] (%d, %d) * (%d, %d)\n", dimsA.x, dimsA.y, dimsA.y,
         dimsB.y);

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block
  unsigned int gridRows = (dimsC.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (dimsC.y + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows); // blocks per grid

  cudaEvent_t start, stop;
  CU_CHECK(cudaEventCreate(&start));
  CU_CHECK(cudaEventCreate(&stop));
  CU_CHECK(cudaEventRecord(start, stream));

  // <<<blocks in grid, block size (threads in block), dynamic shared mem,
  // gpu stream to run on>>>
  // NOTE: m = dimsA.x, n = dimsA.y, k = dimsB.y
  MatMulKernel<BLOCK_SIZE><<<gridSize, blockSize, 1, stream>>>(
      A_d, B_d, C_d, dimsA.x, dimsA.y, dimsB.y);

  CU_CHECK(cudaEventRecord(stop, stream));
  CU_CHECK(cudaEventSynchronize(stop));

  float elapsed;
  CU_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  printf("[TIME] Completed in %.4gms.\n", elapsed);

  CU_CHECK(cudaEventDestroy(start));
  CU_CHECK(cudaEventDestroy(stop));

  CU_CHECK(cudaGetLastError());
  CU_CHECK(cudaStreamSynchronize(stream));

  // copy result back to host
  CU_CHECK(
      cudaMemcpyAsync(C_h, C_d, mem_sizeC, cudaMemcpyDeviceToHost, stream));
  CU_CHECK(cudaStreamSynchronize(stream));

  // printf("\nMatrix A (%d x %d):\n", dimsA.x, dimsA.y);
  // for (int i = 0; i < size_A; i++) {
  //   printf("%4.1f ", A_h[i]);
  //   if ((i + 1) % dimsA.x == 0)
  //     printf("\n");
  // }
  //
  // printf("\nMatrix B (%d x %d):\n", dimsB.x, dimsB.y);
  // for (int i = 0; i < size_B; i++) {
  //   printf("%4.1f ", B_h[i]);
  //   if ((i + 1) % dimsB.x == 0)
  //     printf("\n");
  // }
  //
  // printf("\nMatrix C = A * B (%d x %d):\n", dimsC.x, dimsC.y);
  // for (int i = 0; i < size_C; i++) {
  //   printf("%6.2f ", C_h[i]);
  //   if ((i + 1) % dimsC.x == 0)
  //     printf("\n");
  // }

  // free memory
  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
  cudaFreeHost(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaStreamDestroy(stream);
}

int main() {
  std::cout << "[CUDA] Launching matrix multiplication kernel...\n";
  matMul();
  std::cout << "\n[END]" << std::endl;
  return 0;
}

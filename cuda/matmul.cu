#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// ** lowk this file has a lot of dumb comments but that's js me thinking
// through the code
// ** feel free to ignore it

// set up matrices
// cuda malloc on device
// copy matrices from host -> device
// do matmul
// cpy back to host

void matInit(float *mat, int size, int n) {
  for (int i = 0; i < size; i++) {
    mat[i] = n;
  }
}

// TODO: use shared memory to make this faster
template <int BLOCK_SIZE>
__global__ void MatMulKernel(float *A, float *B, float *C, int wA, int wB) {
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

  // NOTE: have to index by row major bc everything is 1D
  float cSum = 0.0f;
  for (int i = 0; i < wA; i++) {
    cSum += A[wA * row + i] * B[wB * i + col];
  }

  C[row * wB + col] = cSum;
  // printf("%f row: %d, col: %d, %f, %f\n ", cSum, row, col, A[wA * row + 1],
  //        B[wB * 1 + col]);
}

void matMul() {
  // set up data
  dim3 dimsA(3, 5);
  dim3 dimsB(5, 3);
  dim3 dimsC(dimsA.x, dimsB.y);

  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int size_C = dimsC.x * dimsC.y;

  unsigned int mem_sizeA = sizeof(float) * size_A;
  unsigned int mem_sizeB = sizeof(float) * size_B;
  unsigned int mem_sizeC = sizeof(float) * size_C;

  float *A_h, *A_d, *B_h, *B_d, *C_h, *C_d;

  // alloc pinned memory on host for faster cpy times
  cudaMallocHost(&A_h, mem_sizeA);
  cudaMallocHost(&B_h, mem_sizeB);
  cudaMallocHost(&C_h, mem_sizeC);

  matInit(A_h, size_A, 1);
  matInit(B_h, size_B, 2);

  // allocate device mem
  cudaMalloc(&A_d, mem_sizeA);
  cudaMalloc(&B_d, mem_sizeB);
  cudaMalloc(&C_d, mem_sizeC);

  // copy matx's from host to device async
  // NOTE: seperate from host thread, so basically making this nonblocking
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  cudaMemcpyAsync(A_d, A_h, mem_sizeA, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(B_d, B_h, mem_sizeB, cudaMemcpyHostToDevice, stream);

  const int BLOCK_SIZE = 4;

  dim3 blockSize(3, 3); // threads per block
  dim3 gridSize(1, 1);  // blocks per grid
  // <<<blocks in grid, block size (threads in block), dynamic shared mem,
  // gpu stream to run on>>>
  // NOTE: we're passing in dimsA & B .y b/c in cuda width is columns
  MatMulKernel<BLOCK_SIZE>
      <<<gridSize, blockSize, 1, stream>>>(A_d, B_d, C_d, dimsA.y, dimsB.y);

  cudaStreamSynchronize(stream);

  // copy result back to host
  cudaMemcpyAsync(C_h, C_d, mem_sizeC, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  printf("\nMatrix A (%d x %d):\n", dimsA.y, dimsA.x);
  for (int i = 0; i < size_A; i++) {
    printf("%4.1f ", A_h[i]);
    if ((i + 1) % dimsA.x == 0)
      printf("\n");
  }

  printf("\nMatrix B (%d x %d):\n", dimsB.y, dimsB.x);
  for (int i = 0; i < size_B; i++) {
    printf("%4.1f ", B_h[i]);
    if ((i + 1) % dimsB.x == 0)
      printf("\n");
  }

  printf("\nMatrix C = A * B (%d x %d):\n", dimsC.y, dimsC.x);
  for (int i = 0; i < size_C; i++) {
    printf("%6.2f ", C_h[i]);
    if ((i + 1) % dimsC.x == 0)
      printf("\n");
  }

  // free memory
  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
  cudaFreeHost(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  std::cout << "[CUDA] Launching matrix multiplication kernel...";
  matMul();
  std::cout << "\nFinished!" << std::endl;
  return 0;
}

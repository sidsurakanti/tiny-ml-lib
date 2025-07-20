#include "errors.cuh"
#include "matmul.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// ** lowk this file has a lot of dumb comments but that's js me thinking
// ** feel free to ignore it it's a whole yap city down there
// also prolly the messiest file i've ever written

// BIG IDEA:
// set up matrices
// cuda malloc on device
// copy matrices from host -> device
// do matmul
// cpy back to host

const bool DEBUG = false;
const bool CPU = false;
const int BLOCK_SIZE = 16;

void matInit(float *mat, int size, int n) {
  for (int i = 0; i < size; i++) {
    mat[i] = n;
  }
}

void printMat(float *mat, int size, int colWidth) {
  for (int i = 0; i < size; i++) {
    printf("%6.2f ", mat[i]);
    if ((i + 1) % colWidth == 0)
      printf("\n");
  }
}

void cpuMatMul(float *A, float *B, float *C, int m, int n, int k) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < k; col++) {
      for (int i = 0; i < n; i++) {
        C[row * k + col] += A[row * n + i] * B[k * i + col];
      }
    }
  }
}

__global__ void ReluKernel(float *mat, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row >= m || col >= n)
    return;

  float &value = mat[row * n + col];
  value = value > 0 ? value : 0;
}

__global__ void VecMatAddKernel(float *vec, float *mat, int m, int n) {
  int row = threadIdx.y + (blockIdx.y * blockDim.y);
  int col = threadIdx.x + (blockIdx.x * blockDim.x);

  if (row >= m || col >= n)
    return;

  mat[row * n + col] += vec[col];
}

// make sure vec and mat are 1d row majored otherwise you're cooked bro
void vecMatAdd(float *vec, float *mat, int m, int n, int k) {
  if (k != m)
    throw std::runtime_error("Dims mismatch, rows of vec and matrix have to be "
                             "equal (you're ngmi if you keep doing ts bruv "
                             "come back with proper inputs)");
  dim3 blockDim(32, 32);
  dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y) / blockDim.y);

  VecMatAddKernel<<<gridDim, blockDim>>>(vec, mat, m, n);
}

__global__ void BasicMatMulKernel(float *A, float *B, float *C, int m, int n,
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
  // NOTE: have to index by row major bc everything is 1D
  for (int i = 0; i < n; i++) {
    sum += A[row * n + i] * B[k * i + col];
  }

  C[row * k + col] = sum;
}

// (m, n) * (n, k) = (m, k)
template <int block_size>
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n,
                             int k) {
  int offsetY = blockDim.y * blockIdx.y;
  int offsetX = blockDim.x * blockIdx.x;
  int row = threadIdx.y + offsetY;
  int col = threadIdx.x + offsetX;

  // clang-format off
  //
  // NOTE: BIG IDEA:
  // we load partial 16x16 tiles of A and B into shared memory for each element of C contained in the 16x16 thread block; 
  // then we perform partial mat mul;
  // we keep doing this until we can iterate thru all (n) rows in A and (n) cols in B; 
  // at which point we'll have fully completed the matrix mul for a 16x16 block of C;
  //
  // ===========
  // PSUEDO CODE # i dont think anyone except me can understand the bullshit below this line
  //
  // tile iters = ceil(n/blocksize)
  //
  // each thread (an element of C) of the 16x16 block loads
  // their row/col to shared mem: 
  // a[1d row idx + (t * block_size + tx)] & 
  // b[k * (t * block_size + ty) + col]
  //
  // sync the threads so that both BLOCKSIZE*BLOCKSIZE blocks of A & B are
  // loaded in shared mem for the SM;
  //
  // perform partial matrix mul
  // tileA[ty][k:0->15] tileB[k:0->15][tx]!
  //
  // add to accum
  // sync threads again

  __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

  int iters = (n + block_size - 1) / block_size;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < iters; t++) {
    // check if the element we're accessing is out of bounds in the matrix;
    if ((block_size * t + tx < n) && (row < m))
      tileA[ty][tx] = A[row * n + (t * block_size + tx)];
    else 
      tileA[ty][tx] = 0.0f;

    if ((block_size * t + ty < n) && (col < k)) 
      tileB[ty][tx] = B[k * (t * block_size + ty) + col];
    else 
      tileB[ty][tx] = 0.0f;

    __syncthreads();

    // perform mat mul
    for (int i = 0; i < BLOCK_SIZE; i++) {
      sum += tileA[ty][i] * tileB[i][tx];
    }

    // sync threads before we pull next tile into shared memory
    __syncthreads();
  }

  // bounds check here also
  if (row < m && col < k) 
    C[row * k + col] = sum;
}


void matMul() {
  // SET UP DATA

  // m, n (A) && n, k (B)
  // dim3 dimsA(8, 5);
  // dim3 dimsB(5, 10);
  // dim3 dimsA(2048, 2048);
  // dim3 dimsB(2048, 2048);
  dim3 dimsA(16384, 8192);
  dim3 dimsB(8192, 32768);
  dim3 dimsC(dimsA.x, dimsB.y);

  printf("[OPERATION] (%d, %d) * (%d, %d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

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

  unsigned int gridRows = (dimsC.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridCols = (dimsC.y + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(gridCols, gridRows); // blocks per grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // threads per block

  printf("[WARMING UP GPU]\n");
  // warm up runs 
  BasicMatMulKernel<<<gridSize, blockSize, 1, stream>>>(
      A_d, B_d, C_d, dimsA.x, dimsA.y, dimsB.y);
  MatMulKernel<BLOCK_SIZE><<<gridSize, blockSize, 1, stream>>>(
      A_d, B_d, C_d, dimsA.x, dimsA.y, dimsB.y);

  CU_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  float elapsed;
  CU_CHECK(cudaEventCreate(&start));
  CU_CHECK(cudaEventCreate(&stop));

  printf("[RUNNING] SlowMatMul\n");
  CU_CHECK(cudaEventRecord(start, stream));

  // <<<blocks in grid, block size (threads in block), dynamic shared mem,
  // gpu stream to run on>>>
  // NOTE: m = dimsA.x, n = dimsA.y, k = dimsB.y
  BasicMatMulKernel<<<gridSize, blockSize, 1, stream>>>(
      A_d, B_d, C_d, dimsA.x, dimsA.y, dimsB.y);

  CU_CHECK(cudaEventRecord(stop, stream));
  CU_CHECK(cudaEventSynchronize(stop));
  CU_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  printf("[TIME] SlowMatMul completed in %.2fms.\n", elapsed);


  printf("[RUNNING] FastMatMul\n");
  CU_CHECK(cudaEventRecord(start, stream));

  // <<<blocks in grid, block size (threads in block), dynamic shared mem,
  // gpu stream to run on>>>
  // NOTE: m = dimsA.x, n = dimsA.y, k = dimsB.y
  MatMulKernel<BLOCK_SIZE><<<gridSize, blockSize, 1, stream>>>(
      A_d, B_d, C_d, dimsA.x, dimsA.y, dimsB.y);

  CU_CHECK(cudaEventRecord(stop, stream));
  CU_CHECK(cudaEventSynchronize(stop));
  CU_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  printf("[TIME] FastMatMul completed in %.2fms.\n", elapsed);

  CU_CHECK(cudaEventDestroy(start));
  CU_CHECK(cudaEventDestroy(stop));

  // wait host thread & error check
  CU_CHECK(cudaGetLastError());
  CU_CHECK(cudaStreamSynchronize(stream));

  // copy result back to host
  CU_CHECK(
      cudaMemcpyAsync(C_h, C_d, mem_sizeC, cudaMemcpyDeviceToHost, stream));
  CU_CHECK(cudaStreamSynchronize(stream));
  
  // free device memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaStreamDestroy(stream);

  if (DEBUG) {
    printf("\n[MAT A] (%dx%d):\n", dimsA.x, dimsA.y);
    printMat(A_h, size_A, dimsA.y);

    printf("\n[MAT B] (%dx%d):\n", dimsB.x, dimsB.y);
    printMat(B_h, size_B, dimsB.y);

    printf("\n[MAT C = A * B] (%dx%d):\n", dimsC.x, dimsC.y);
    printMat(C_h, size_C, dimsC.y);

  }


  if (CPU) {
    float* C = (float*)calloc(size_C, sizeof(float));

    printf("[RUNNING] CpuMatMul\n");

    auto s = std::chrono::high_resolution_clock::now();
    cpuMatMul(A_h, B_h, C, dimsA.x, dimsA.y, dimsB.y);
    auto e = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
    printf("[TIME] CPU finished in %ldms.\n", duration.count());

    if (DEBUG) printMat(C, size_C, dimsC.y);

    free(C);
  }

  // free cpu memory
  cudaFreeHost(C_h);
  cudaFreeHost(A_h);
  cudaFreeHost(B_h);
}

int main() {
  std::cout << "[CUDA] Launching matrix multiplication kernel...\n";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "[DEVICE INFO]" << std::endl;
  std::cout << "DEVICE: " << prop.name << std::endl;
  std::cout << "SM COUNT: " << prop.multiProcessorCount << std::endl;
  std::cout << "MAX THREADS / BLOCK: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "WARP SIZE: " << prop.warpSize << std::endl;
  std::cout << "SHARED MEMORY / BLOCK: " << prop.sharedMemPerBlock / (1024) << "KB" << std::endl;
  std::cout << "GLOBAL MEMORY: " << prop.totalGlobalMem / (1024*1024*1024) << "GB" << std::endl;

  matMul();

  std::cout << "[END]" << std::endl;
  return 0;
}


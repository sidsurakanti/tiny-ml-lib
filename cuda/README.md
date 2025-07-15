```bash
>>> nvcc -arch=sm_89 matmul.cu -o matmul
```
```bash
>>> ./matmul
[CUDA] Launching matrix multiplication kernel...
[OPERATION] (10, 5) * (5, 7)
[TIME] Completed in 0.262ms.

Matrix A (10 x 5):
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0

Matrix B (5 x 7):
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0

Matrix C = A * B (10 x 7):
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00
 10.00  10.00  10.00  10.00  10.00  10.00  10.00

[END]
```

### RESULTS FROM UPGRADE TO SHARED MEMORY:
- 1.3x speed up with shared memory optimizations 
- 5800x from cpu (10x from numpy)

```bash
>>> ./matmul

[CUDA] Launching matrix multiplication kernel...
[OPERATION] (2048, 2048) * (2048, 2048)
[WARMING UP GPU]
[RUNNING] SlowMatMul
[TIME] SlowMatMul completed in 19.85ms.
[RUNNING] FastMatMul
[TIME] FastMatMul completed in 16.02ms.
[TIME] CPU finished in 92993ms.
[END]
```
```bash
>>> ./matmul

[CUDA] Launching matrix multiplication kernel...
[OPERATION] (16384, 8192) * (8192, 32768)
[WARMING UP GPU]
[RUNNING] SlowMatMul
[TIME] SlowMatMul completed in 9569.36ms.
[RUNNING] FastMatMul
[TIME] FastMatMul completed in 7426.05ms.
[END]
```

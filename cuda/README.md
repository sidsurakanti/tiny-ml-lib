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

RESULTS FROM UPGRADE TO SHARED MEMORY:
1.275x speed up (lowk should be more from what i've read i'm not sure why the speed up was so low)
```bash
>>> ./matmul

[CUDA] Launching matrix multiplication kernel...
[OPERATION] (8193, 4099) * (4099, 16385)
[WARMING UP GPU]
[RUNNING] SlowMatMul
[TIME] SlowMatMul completed in 1253.076ms.
[RUNNING] FastMatMul
[TIME] FastMatMul completed in 982.414ms.
[END]
```


# Overview
## MaxPool
2.5x raw speed up on MaxPooling vs Python
```bash
Input shape: (60000, 784)
Labels shape: (60000,)
Model(
  [0] Conv2d       ((1, 28, 28) → (5, 24, 24))
  [1] ReLU
  [2] MaxPool
  [3] Flatten
  [4] Linear       (720 → 128)
  [5] ReLU
  [6] Linear       (128 → 10)
  Loss: CrossEntropyLoss
  Total parameters: 96,583
  Device: CPU
)

TRAINING...
EPOCH 1/5, Loss: 1.75497549
EPOCH 2/5, Loss: 0.75797579
EPOCH 3/5, Loss: 0.53465346
EPOCH 4/5, Loss: 0.45654565
EPOCH 5/5, Loss: 0.41454145
Finished in: 492.62s # old time was 1200s

EVALUATING...
Sample labels: [8 5 6 4 2 4 2 4 1 3]
Sample preds: [8 5 6 4 4 4 2 4 1 3]
Accuracy: 89.95%

Save weights? (y/n) >>> n
```
## Matrix Mul
- 1.3x speed up with shared memory optimizations 
- 5800x from cpu (10x from numpy)
```bash
>>> nvcc -arch=sm_89 matmul.cu -o matmul
```
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


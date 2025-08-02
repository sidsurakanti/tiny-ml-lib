# Overview

A lightweight deep learning framework built from scratch using raw CUDA & Python with functionality for core components like CNN and MLPs.

## Features

- Fully connected layer
- GPU Acceleration (15x faster than regular NumPy, same speed as PyTorch for smaller stuff)
- Convolutional layer
- Flatten layer
- Max pooling layer
- ReLU activation
- Softmax layer
- Model save/load
- Cross Entropy Loss & MSE Loss
- Model & sequential classes
- Training & eval loop
- Mini-batching

## Usage
Convolutional NN  on MNIST
```bash
>>> py main.py
Input shape: (60000, 784)
Labels shape: (60000,)
Model(
  [0] Conv2d       ((1, 28, 28) → (5, 24, 24))
  [1] ReLU
  [2] Flatten
  [3] Linear       (2880 → 128)
  [4] ReLU
  [5] Linear       (128 → 10)
  Loss: CrossEntropyLoss
  Total parameters: 373,063
)

TRAINING...
EPOCH 1/10, Loss: 0.1227
...
EPOCH 9/10, Loss: 0.0355
EPOCH 10/10, Loss: 0.0347
Time spent training: 437.89s

EVALUATING...
Sample labels: [9 2 9 8 9 7 1 2 4 3]
Sample preds: [9 2 9 8 9 7 1 2 4 3]
Accuracy: 98.32%

Save weights? (y/n) >>> y
File name? (empty for default) >>> cnn-weights
Saved model weights to cnn-weights.pkl
```
With MaxPool (didn't bother running for longer than 5 epochs cus it's slow asf)
```bash
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
)

TRAINING...
EPOCH 1/5, Loss: 0.6285
EPOCH 2/5, Loss: 0.4906
EPOCH 3/5, Loss: 0.3883
EPOCH 4/5, Loss: 0.3109
EPOCH 5/5, Loss: 0.3366
Time spent training: 1116.21s

EVALUATING...
Sample labels: [4 9 5 0 6 0 7 9 8 8]
Sample preds: [4 9 5 0 6 0 9 9 8 8]
Accuracy: 87.67%
```

MLP on MNIST (GPU)
```bash
>>> py main.py
Input shape: (60000, 784)
Labels shape: (60000,)
Model(
  [0] Linear       (784 → 512)
  [1] ReLU
  [2] Linear       (512 → 512)
  [3] ReLU
  [4] Linear       (512 → 512)
  [5] ReLU
  [6] Linear       (512 → 10)
  Loss: CrossEntropyLoss
  Total parameters: 932,362
  Device: GPU
)

TRAINING...
EPOCH 1/10, Loss: 0.5499499
EPOCH 2/10, Loss: 0.4014014
EPOCH 3/10, Loss: 0.3486486
EPOCH 4/10, Loss: 0.3164164
EPOCH 5/10, Loss: 0.2933933
EPOCH 6/10, Loss: 0.2757757
EPOCH 7/10, Loss: 0.2610610
EPOCH 8/10, Loss: 0.2491491
EPOCH 9/10, Loss: 0.2388388
EPOCH 10/10, Loss: 0.229797
Finished in: 9.51s

EVALUATING...
Sample labels: [7 3 1 1 0 8 0 8 6 4]
Sample preds: [7 3 1 1 0 0 0 8 6 4]
Accuracy: 95.70%

Save weights? (y/n) >>> n
```

With pretrained weights: 

```bash
Loaded model weights from mlp-weights.pkl

EVALUATING...
Sample labels: [2 0 1 9 6 5 5 6 7 8]
Sample preds: [2 0 1 9 6 5 5 6 7 8]
Accuracy: 98.13%
```

# Benchmarks
> ⚠️ **Disclaimer**  
> This library doesn't have autograd (yet), graph tracing, mixed precision, cuDNN, cuBLAS, or any of the fancy stuff PyTorch does.  
> It only runs "faster" because it's lightweight
> Still beats pytorch at batch sizes < 512 for MNIST though, so it's a win in my book.

All benchmarks were run on a **RTX 4060**, training a simple MNIST NN from scratch using this library’s GPU backend.

Model:
`Linear(784 → 512) → ReLU → Linear(512 → 512) → ReLU → Linear(512 → 512) → ReLU → Linear(512 → 10)`  
Loss: CrossEntropy
Optimizer: SGD, `lr=0.1`
Epochs: 10

| Batch Size | Framework | Time (10 Epochs) |
|------------|-----------|------------------|
| 64         | PyTorch   | 27.2s            |
| 64         | This lib  | **20.2s**        |
| 512        | PyTorch   | 9.7s             |
| 512        | This lib  | **10.0s**        |


## Why make this?

- I wanted to learn more about ml libraries and autograd (didn't get to implementing it) 
- Wanted to implement a Convolutional layer on my own
- Wanted to experiment with a framework and learn cool stuff
- Didn't want to see that ugly ahh blue python color in the repo badge so I remade it in cuda too 


## Stack
![CUDA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)  
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- CUDA Toolkit 
- CMake
- gcc or g++

### Installation

Clone the repo:

```bash
git clone https://github.com/sidsurakanti/tiny-ml-lib.git
cd repo
```

Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build core cuda lib
```bash
mkdir build && cd build
cmake .. && make && make install
cd ..
```

Run the app

```bash
python3 main.py
```

or

```bash
python main.py
```

## Roadmap

- [x] MLP basic functionality
- [x] Add Conv2d
- [x] Add pooling layer
- [x] Add weight inits
- [ ] Add more activations etc
- [x] Cuda remake

## Support

Need help? Ping me on [discord](https://discord.com/users/521872289231273994)

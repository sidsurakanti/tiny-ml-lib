# Overview

A lite machine learning framework built from scratch using NumPy with functionality for core components like CNN and MLPs.

## Features

- Fully connected layer
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
Accuracy: 97.52%

Save weights? (y/n) >>> y
File name? (empty for default) >>> cnn-weights
Saved model weights to cnn-weights.pkl
```
...again with MaxPool
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

MLP on MNIST
```bash
Input shape: (60000, 784)
Labels shape: (60000,)
Model(
  [0] Linear       (784 → 256)
  [1] ReLU
  [2] Linear       (256 → 256)
  [3] ReLU
  [4] Linear       (256 → 10)
  Loss: CrossEntropyLoss
  Total parameters: 269,322
)

TRAINING...
EPOCH 1/25, Loss: 0.4248
EPOCH 2/25, Loss: 0.2641
EPOCH 3/25, Loss: 0.1849
...
EPOCH 25/25, Loss: 0.0451
Time spent training: 31.27s

EVALUATING...
Sample labels: [1 6 9 6 8 8 3 1 4 2]
Sample preds: [1 6 9 6 8 8 3 1 4 2]
Accuracy: 98.20%

Save weights? (y/n) >>> y
File name? (empty for default) >>> mlp-weights
Saved model weights to mlp-weights.pkl
```
```bash
Loaded model weights from mlp-weights.pkl

EVALUATING...
Sample labels: [2 0 1 9 6 5 5 6 7 8]
Sample preds: [2 0 1 9 6 5 5 6 7 8]
Accuracy: 98.13%
```


## Why make this?

- I wanted to learn more about ml libraries and autograd (didn't get to implementing it) 
- Wanted to implement a Convolutional layer on my own
- Wanted to experiment with a framework and learn cool stuff


## Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)  
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

Clone the repo:

```bash
git clone https://github.com/sidsurakanti/tiny-ml-lib.git
cd repo
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
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
- [ ] More loss func's
- [ ] C++ remake

## Support

Need help? Ping me on [discord](https://discord.com/users/521872289231273994)

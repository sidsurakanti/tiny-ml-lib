# Overview

A small neural network library built from scratch using NumPy. Designed for learning and experimentation, this project walks through implementing core ML components (layers, activation functions, loss functions, and training logic) without external ML frameworks.

## Features

- Fully connected layer
- Convolutional layer
- Flatten layer
- ReLU activation
- Cross Entropy Loss & MSE Loss
- Model & sequential classes
- Data mini-batching

## Usage
Convolutional neural network on MNIST
```bash
Input shape: (60000, 784)
Labels shape: (60000,)

ARCHITECTURE:
<Conv2D: (1, 28, 28) -> (5, 24, 24)>
<ReLU>
<Flatten>
<Linear: 2880 -> 128>
<ReLU>
<Linear: 128 -> 10>
<CrossEntropyLoss>

TRAINING
EPOCH 1/5, Loss: 1.64071.6407
EPOCH 2/5, Loss: 1.12811.1281
EPOCH 3/5, Loss: 0.69510.6951
EPOCH 4/5, Loss: 0.57120.5712
EPOCH 5/5, Loss: 0.33620.3362

EVALUATING
Sample labels: [6 8 1 9 8 0 8 1 1 2]
Sample preds: [6 8 1 9 8 0 8 1 1 2]
Accuracy: 85.07%
```

MLP on MNIST
```bash
Input shape: (60000, 784)
Labels shape: (60000,)

ARCHITECTURE:
<Linear: 784 -> 128>
<ReLU: 128 -> 128>
<Linear: 128 -> 64>
<ReLU: 64 -> 64>
<Linear: 64 -> 10>
<CrossEntropyLoss>

TRAINING
EPOCH 1/50, Loss: 6.3107107
EPOCH 2/50, Loss: 5.5198198
EPOCH 3/50, Loss: 4.4785785
EPOCH 4/50, Loss: 4.2003003
EPOCH 5/50, Loss: 3.8099099
EPOCH 6/50, Loss: 3.5367367
EPOCH 7/50, Loss: 2.5029029
EPOCH 8/50, Loss: 2.3987987
EPOCH 9/50, Loss: 2.1766766
EPOCH 10/50, Loss: 2.145050
...
EPOCH 47/50, Loss: 0.15611561
EPOCH 48/50, Loss: 0.17551755
EPOCH 49/50, Loss: 0.14681468
EPOCH 50/50, Loss: 0.15551555

EVALUATING
Sample labels: [1 9 3 8 6 1 4 1 7 5]
Sample preds: [1 9 3 8 8 1 4 1 7 5]
Accuracy: 95.09%
```

## Why make this?

- I wanted to learn more about ml libraries and how they work under the hood
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
- [ ] Add Conv2d
- [ ] Add weight inits
- [ ] More loss func's
- [ ] C++ remake

## Support

Need help? Ping me on [discord](https://discord.com/users/521872289231273994)

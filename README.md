# Overview

A minimal neural network library built from scratch using NumPy. Designed for learning and experimentation, this project walks through implementing core ML components (layers, activation functions, loss functions, and training logic) without external ML frameworks.

## Features

- Fully connected layer
- ReLU activation
- Cross Entropy Loss & MSE Loss
- Model & sequential classes
- Data mini-batching

```bash
Input shape: (784, 60000)
Labels shape: (60000,)

ARCHITECTURE:
<Linear: 784 -> 128>
<ReLU: 128 -> 128>
<Linear: 128 -> 64>
<ReLU: 64 -> 64>
<Linear: 64 -> 10>
<CrossEntropyLoss>

TRAINING
EPOCH 1/50, Loss: 2.3107107
EPOCH 2/50, Loss: 2.5198198
EPOCH 3/50, Loss: 2.4785785
EPOCH 4/50, Loss: 2.2003003
EPOCH 5/50, Loss: 1.8099099
EPOCH 6/50, Loss: 1.5367367
EPOCH 7/50, Loss: 1.5029029
EPOCH 8/50, Loss: 1.3987987
EPOCH 9/50, Loss: 1.1766766
EPOCH 10/50, Loss: 1.145050
...
EPOCH 48/50, Loss: 0.268282
EPOCH 49/50, Loss: 0.259898
EPOCH 50/50, Loss: 0.258080

EVALUATING
Sample preds: [5 0 4 1 9 2 1 3 1 4]
Sample labels: [5 0 4 1 9 2 1 3 1 4]
Accuracy: 92.90% 
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

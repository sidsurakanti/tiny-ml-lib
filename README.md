# Overview

A small machine learning library built from scratch using NumPy with functionality for core components like CNN and MLPs.

## Features

- Fully connected layer
- Convolutional layer
- Flatten layer
- ReLU activation
- Cross Entropy Loss & MSE Loss
- Model & sequential classes
- Training & eval loop
- Mini-batching

## Usage
Convolutional NN  on MNIST
```bash
Input shape: (60000, 784)
Labels shape: (60000,)

ARCHITECTURE:
<Conv2D: (1, 28, 28) -> (5, 24, 24), Filters: 5, 5x5>
<ReLU>
<Flatten>
<Linear: 2880 -> 128>
<ReLU>
<Linear: 128 -> 10>
<CrossEntropyLoss>

TRAINING...
EPOCH 1/15, Loss: 1.0956
EPOCH 2/15, Loss: 0.7308
EPOCH 3/15, Loss: 0.4557
EPOCH 4/15, Loss: 0.4566
EPOCH 5/15, Loss: 0.4934
EPOCH 6/15, Loss: 0.3920
EPOCH 7/15, Loss: 0.3638
EPOCH 8/15, Loss: 0.4017
EPOCH 9/15, Loss: 0.3439
...
EPOCH 24/25, Loss: 0.1281
EPOCH 25/25, Loss: 0.0886
Time spent training: 1122.34s

EVALUATING...
Sample labels: [3 6 0 0 9 1 8 2 7 3]
Sample preds: [3 2 0 0 9 1 8 2 7 3]
Accuracy: 94.77%
```

MLP on MNIST
```bash
Input shape: (60000, 784)
Labels shape: (60000,)

ARCHITECTURE:
<Linear: 784 -> 128>
<ReLU>
<Linear: 128 -> 64>
<ReLU>
<Linear: 64 -> 10>
<CrossEntropyLoss>

TRAINING...
EPOCH 1/50, Loss: 6.3107
EPOCH 2/50, Loss: 5.5198
EPOCH 3/50, Loss: 4.4785
EPOCH 4/50, Loss: 4.2003
EPOCH 5/50, Loss: 3.8099
EPOCH 6/50, Loss: 3.5367
EPOCH 7/50, Loss: 2.5029
EPOCH 8/50, Loss: 2.3987
EPOCH 9/50, Loss: 2.1766
EPOCH 10/50, Loss: 2.1450
...
EPOCH 47/50, Loss: 0.0691
EPOCH 48/50, Loss: 0.0742
EPOCH 49/50, Loss: 0.0860
EPOCH 50/50, Loss: 0.0722
Time spent training: 29.86s

EVALUATING...
Sample labels: [6 2 2 7 6 0 6 9 4 8]
Sample preds: [6 2 2 7 6 0 6 9 4 8]
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
- [x] Add Conv2d
- [ ] Add weight inits
- [ ] More loss func's
- [ ] C++ remake

## Support

Need help? Ping me on [discord](https://discord.com/users/521872289231273994)

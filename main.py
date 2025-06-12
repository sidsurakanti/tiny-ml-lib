from model import Model
from defs import Sequence, Array
from linear import Linear
from conv2d import Conv2d
from flatten import Flatten
from relu import ReLU
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from losses import CrossEntropyLoss, MSELoss

# MNIST
# just download mnist into /dataset/* and change file names to match
test = pd.read_csv("./dataset/train.csv")
train = pd.read_csv("./dataset/train.csv")

test = np.array(test.T)
train = np.array(train.T)

y_test, X_test = test[0], test[1:] / 255
y_train, X_train = train[0], train[1:] / 255

X_test, X_train = X_test.T, X_train.T
m, n = X_train.shape
cX_train = X_train.reshape(m, 1, 28, 28)
cX_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print("Input shape:", X_train.shape)
print("Labels shape:", y_train.shape)

plt.imshow(X_train[np.random.randint(m)].reshape((28, 28)))
# plt.savefig("eg.png")


def main():
  fc = Linear(n, 10) # (10, 784) * (784, m) -> (10, m) 
  a = ReLU() # output-> (10, m)
  fc2 = Linear(10, 10)

  # 28 should acc be sqrt(n) but i dont feel like generalizing
  conv2d = Conv2d((1, 28, 28), 5, 3)

  # print(conv2d)
  # out = conv2d.forward(cX_train)
  # print(out.shape)
  # dz = np.ones_like(out)
  # dx = conv2d.backwards(dz)  
  # print(dx.shape)
  # print(dx[0])

  loss_fn = CrossEntropyLoss()
  # loss_fn = MSELoss()

  # sequence = [
  #       Linear(784, 128),
  #       ReLU(),
  #       Linear(128, 64),
  #       ReLU(),
  #       Linear(64, 10)
  #     ]

  sequence = [
        Conv2d((1, 28, 28), 5, 5),
        ReLU(),
        Flatten(),
        Linear(24*24*5, 128),
        ReLU(),
        Linear(128, 10)
      ]

  print("\nARCHITECTURE:")
  for layer in sequence:
    print(layer)
  print(loss_fn)

  model = Model(sequence, loss_fn)

  print("\nTRAINING")
  # model(50, X_train, y_train, batch_size=32)
  model(5, cX_train, y_train, batch_size=32)

  print("\nEVALUATING")
  # acc = model.evaluate(X_test, y_test)
  acc = model.evaluate(cX_test, y_test)
  print(f"Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
  main()

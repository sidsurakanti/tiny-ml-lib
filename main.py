from model import Model
from defs import Sequence, Array
from linear import Linear
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
n, m = X_train.shape
print("Input shape:", X_train.shape)
print("Labels shape:", y_train.shape)

plt.imshow(X_train[:, np.random.randint(m)].reshape((28, 28)))
# plt.savefig("eg.png")

def main():
  fc = Linear(n, 10) # (10, 784) * (784, m) -> (10, m) 
  a = ReLU(10, 10) # output-> (10, m)
  fc2 = Linear(10, 10)

  loss_fn = CrossEntropyLoss()
  # loss_fn = MSELoss()
  print("\nARCHITECTURE:")
  print(fc)
  print(a)
  print(fc2)
  print(loss_fn, "\n") 

  # print("\nFORWARD PASS")
  # out = fc.forward(X_train) 
  # print("FC output:", out.shape)
  # out = a.forward(out)
  # print("Activation output:", out.shape)
  # out = fc2.forward(out)
  # print("FC2 output:", out.shape)
  # loss = loss_fn.loss(out, y_train)
  # print("Loss:", loss)
  
  # print("\nBACKWARDS PASS")
  # dA2 = loss_fn.backwards()
  # print("Loss grads:", dA2.shape)
  # dZ2 = fc2.backwards(dA2)
  # print("FC2 grads:", fc2.dW.shape, fc2.db.shape)
  # dA = a.backwards(dZ2)
  # print("Activation grads:", dA.shape)
  # dZ = fc.backwards(dA)
  # print("FC grads:", fc.dW.shape, fc.db.shape)
  # # print(np.unique(dW, axis=1))
  # # print(np.min(dW), np.max(dW))
  # # print(np.ptp(dW))

  sequence = [
        Linear(784, 128),
        ReLU(128, 128),
        Linear(128, 64),
        ReLU(64, 64),
        Linear(64, 10)
      ]


  # model = Model([fc, a, fc2], loss_fn)
  model = Model(sequence, loss_fn)

  print("TRAINING")
  model(50, X_train, y_train, batch_size=128)

  print("EVALUATING")
  acc = model.evaluate(X_test, y_test)
  print(f"Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
  main()

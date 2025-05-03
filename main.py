from linear import Linear
from relu import ReLU
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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

plt.imshow(X_train[:, np.random.randint(m)].reshape((28, 28)))
# plt.savefig("eg.png")

def main():
  fc = Linear(n, 10) # (10, 784) * (784, m) -> (10, m) 
  a = ReLU(10, 10) # output-> (10, m)
  print("\nARCHITECTURE:")
  print(fc)
  print(a)

  print("\nFORWARD PASS")
  out = fc.forward(X_train) 
  print("FC output:", out.shape)
  out = a(out)
  print("Activation output:", out.shape)
  
  print("\nBACKWARDS PASS")
  dA = a.backwards(1)
  print("Activation grads:", dA.shape)
  dW, db = fc.backwards(dA)
  print("FC grads:", dW.shape, db.shape)
  # print(np.unique(dW, axis=1))
  # print(np.min(dW), np.max(dW))
  # print(np.ptp(dW))

if __name__ == "__main__":
  main()

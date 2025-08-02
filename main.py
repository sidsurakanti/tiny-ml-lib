from model import Model
from defs import Sequence
from linear import Linear
from conv2d import Conv2d
from maxpool import MaxPool
from flatten import Flatten
from relu import ReLU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from losses import CrossEntropyLoss, MSELoss

np.set_printoptions(linewidth=200)

# MNIST
# just download mnist into /dataset/* and change file names to match
test = pd.read_csv("./dataset/test.csv")
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

# plt.imshow(X_train[np.random.randint(m)].reshape((28, 28)))
# plt.savefig("eg.png")


def main():
    BATCH_SIZE = 64

    # sequence = [
    #     Conv2d((1, 28, 28), 5, 5),
    #     ReLU(),
    #     # MaxPool(),
    #     Flatten(),
    #     Linear(24 * 24 * 5, 128),
    #     ReLU(),
    #     Linear(128, 10),
    # ]

    sequence: Sequence = [
        Linear(784, 512),
        ReLU(512),
        Linear(512, 512),
        ReLU(512),
        Linear(512, 512),
        ReLU(512),
        Linear(512, 10),
    ]

    loss_fn = CrossEntropyLoss()  # MSELoss()

    model = Model(sequence, loss_fn)
    # model.load("mlp-weights.pkl")
    model.toGPU(BATCH_SIZE)
    model(10, X_train, y_train, learning_rate=3e-2, batch_size=BATCH_SIZE)
    # model(10, cX_train, y_train, batch_size=32)

    acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    # acc = model.evaluate(cX_test, y_test)
    print(f"Accuracy: {acc*100:.2f}%")

    if input("\nSave weights? (y/n) >>> ").lower() in ("y", "yes"):
        fpath = input("File name? (empty for default) >>> ")
        if fpath.strip():
            model.save(fpath + ".pkl")
        else:
            model.save()


if __name__ == "__main__":
    main()

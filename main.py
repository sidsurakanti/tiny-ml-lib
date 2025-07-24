from model import Model
from defs import Sequence, Array
from linear import Linear
from conv2d import Conv2d
from maxpool import MaxPool
from flatten import Flatten
from relu import ReLU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from losses import CrossEntropyLoss, MSELoss
from native import toCPU

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

# print("Input shape:", X_train.shape)
# print("Labels shape:", y_train.shape)

plt.imshow(X_train[np.random.randint(m)].reshape((28, 28)))
# plt.savefig("eg.png")


def main():
    sequence = [
        Linear(784, 1024),
        ReLU(1024),
        Linear(1024, 1024),
        ReLU(1024),
        Linear(1024, 1024),
        ReLU(1024),
        Linear(1024, 10),
    ]

    conv2d = Conv2d((1, 28, 28), 5, 3)
    # out = conv2d.forward(cX_train)
    # print(out.shape)
    # dz = np.ones_like(out)
    # dx = conv2d.backwards(dz)
    # print(dx.shape, np.mean(dx), dx[0])

    # fc = Linear(784, 10)  # (m, 784) * (784, 10) -> (m, 10)
    # a = ReLU(10)  # output -> (m, 10)
    # fc2 = Linear(10, 10)
    loss_fn = CrossEntropyLoss()
    # BATCH_SIZE = 10
    #
    # fc.toGPU(BATCH_SIZE)
    # a.toGPU(BATCH_SIZE)
    # fc2.toGPU(BATCH_SIZE)
    # loss_fn.toGPU()
    #
    # out = fc.forward(X_train[:BATCH_SIZE])
    # out = a.forward(out)
    # out = fc2.forward(out)
    #
    # print("FORWARD LOGS")
    # print("==========================")
    #
    # fc.debug()
    # fc2.debug()
    #
    # outH = toCPU(out, fc2.batch_size, fc2.output_shape)
    # loss = loss_fn(outH, y_train[:BATCH_SIZE], out)
    #
    # print("BACKWARDS LOGS")
    # print("==========================")
    #
    # dZ = loss_fn.backwards()
    # print(f"loss gradient norm: {np.linalg.norm(toCPU(dZ, BATCH_SIZE, 10))}")
    #
    # dZ = fc2.backwards(dZ)
    # print(f"after fc2 norm: {np.linalg.norm(toCPU(dZ, BATCH_SIZE, 10))}")
    #
    # dZ = a.backwards(dZ)
    # print(f"after relu1 norm: {np.linalg.norm(toCPU(dZ, BATCH_SIZE, 10))}")
    #
    # dZ = fc.backwards(dZ)
    #
    # print()
    #
    # fc.debug()
    # a.debug()
    # fc2.debug()
    # # loss_fn
    #
    # lr = 0.03
    #
    # w1_before = toCPU(fc.gpuPtrs[0][0], fc.input_shape, fc.output_shape)
    # w2_before = toCPU(fc2.gpuPtrs[0][0], fc2.input_shape, fc2.output_shape)
    #
    # dw1_before = toCPU(fc.gpuPtrs[1][0], fc.input_shape, fc.output_shape)
    # dw2_before = toCPU(fc2.gpuPtrs[1][0], fc2.input_shape, fc2.output_shape)
    #
    # print()
    # print("Layer 1 weights before:", w1_before[0, :5])
    # print("Layer 2 weights before:", w2_before[0, :5])
    #
    # print("Layer 1 grad:", dw1_before[0, :5])
    # print("Layer 2 grad:", dw2_before[0, :5])
    # fc.step(lr)
    # fc2.step(lr)
    #
    # w1_after = toCPU(fc.gpuPtrs[0][0], fc.input_shape, fc.output_shape)
    # w2_after = toCPU(fc2.gpuPtrs[0][0], fc2.input_shape, fc2.output_shape)
    #
    # print("Layer 1 change:", w1_after[0, :5] - w1_before[0, :5])
    # print("Layer 2 change:", w2_after[0, :5] - w2_before[0, :5])
    # print(loss)

    MSELoss()

    # sequence = [
    #     Conv2d((1, 28, 28), 5, 5),
    #     ReLU(),
    #     # MaxPool(),
    #     Flatten(),
    #     Linear(24 * 24 * 5, 128),
    #     ReLU(),
    #     Linear(128, 10),
    # ]

    model = Model(sequence, loss_fn)
    # model.load("mlp-weights.pkl")
    model.toGPU()

    model(10, X_train, y_train, learning_rate=0.03, batch_size=512)
    # model(10, cX_train, y_train, batch_size=32)

    acc = model.evaluate(X_test, y_test)
    # acc = model.evaluate(cX_test, y_test)
    print(f"Accuracy: {acc*100:.2f}%")
    #
    if input("\nSave weights? (y/n) >>> ").lower() in ("y", "yes"):
        fpath = input("File name? (empty for default) >>> ")
        if fpath.strip():
            model.save(fpath + ".pkl")
        else:
            model.save()


if __name__ == "__main__":
    main()

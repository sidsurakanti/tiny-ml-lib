import numpy as np


class NativeFallback:
    def __init__(self):
        print("[warn] CUDA extension not found. Using NumPy fallback backend.")

    def toGPU(self, x):
        return x  # no-op

    def toCPU(self, x):
        return x  # no-op

    def updateGpuMemory(self, *args, **kwargs):
        pass

    def initBuff(self, *args, **kwargs):
        pass

    def initBuffers(self, *args, **kwargs):
        pass

    def matMatSub(self, a, b):
        return a - b

    def relu(self, x):
        return np.maximum(0, x)

    def reluBack(self, grad, x):
        return grad * (x > 0)

    def linear(self, x, w, b=None):
        y = x @ w.T
        if b is not None:
            y += b
        return y

    def linearBack(self, grad, x, w):
        dx = grad @ w
        dw = grad.T @ x
        db = grad.sum(axis=0)
        return dx, dw, db

    def matmul(self, a, b):
        return a @ b

    def maxpool(self, x, pool_size=2, stride=2):
        # dumb 2D maxpool fallback
        n, h, w, c = x.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        out = np.zeros((n, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                patch = x[
                    :,
                    i * stride : i * stride + pool_size,
                    j * stride : j * stride + pool_size,
                    :,
                ]
                out[:, i, j, :] = patch.max(axis=(1, 2))
        return out


native = NativeFallback()

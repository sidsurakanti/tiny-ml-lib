import numpy as np
from defs import Array
from layer import Layer
from math import ceil
from native import maxpool


class MaxPool(Layer):
    def __init__(self) -> None:
        self._X = None
        self.mask = np.zeros((0, 0))
        self.arg_idxs = []
        self.mask_idxs = np.empty(0, np.int64)  # for gpu use
        self._onGPU = False

    def forward(self, X: Array):
        self.X = X

        # presume x is of (5, 3, 28, 28)
        filters, channels, h, w = self.X.shape

        if self._onGPU:
            res, idxs = maxpool(X.reshape(-1), filters, channels, h, w, 2, 2, True)
            self.mask_idxs = idxs
            return res

        nh, nw = ceil(h / 2), ceil(w / 2)
        res = np.zeros((filters, channels, nh, nw))

        # padded = np.pad(self.X, [(0, 0), (0, 0), (0, h%2), (0, w%2)], mode='constant')

        # can prob do this all w np but yea
        for f in range(filters):
            for c in range(channels):
                view = self.X[f, c]  # 28x28

                for i in range(nh):
                    for j in range(nw):
                        i0, i1 = i * 2, min((i + 1) * 2, h)
                        j0, j1 = j * 2, min((j + 1) * 2, w)

                        pooler = view[i0:i1, j0:j1]
                        res[f, c, i, j] = np.max(pooler)

                        # set the backwards mask while at it
                        idx = np.unravel_index(np.argmax(pooler), pooler.shape)
                        self.arg_idxs.append([f, c, i0 + idx[0], j0 + idx[1]])
                        # self.mask[f, c, i0 + idx[0], j0 + idx[1]] = 1

        return res

    def backwards(self, dZ) -> Array:
        if self._onGPU:
            coords = np.unravel_index(self.mask_idxs.flatten(), self.mask.shape)
            self.mask[coords] = dZ.flatten()
        else:
            idxs = np.array(self.arg_idxs).T
            self.mask[tuple(idxs)] = dZ.flatten()
            self.arg_idxs = []  # reset

        return self.mask

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, val):
        self.mask = np.zeros_like(val)
        self._X = val

    def toGPU(self):
        self._onGPU = True
        # assert False, "No GPU implementation for Convolutional Layer yet"

    def __repr__(self):
        return f"<MaxPool>"


if __name__ == "__main__":
    r = 3
    xyz = np.random.randn(1, 3, r, r)
    print(xyz)
    pool = MaxPool()
    res = pool.forward(xyz)
    print(res)
    dz = np.ones_like(res)
    dx = pool.backwards(dz)
    print(dx)

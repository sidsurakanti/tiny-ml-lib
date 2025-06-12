import numpy as np
from layer import Layer 

class Flatten(Layer):
    def forward(self, X):
        self.old_shape = X.shape
        res = X.reshape(X.shape[0], -1)
        # print(res.shape)
        return res

    def backwards(self, out):
        return out.reshape(self.old_shape)

    def __repr__(self):
        return f"<Flatten>"



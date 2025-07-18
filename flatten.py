from layer import Layer
from defs import Array


class Flatten(Layer):
    def forward(self, X) -> Array:
        self.old_shape = X.shape
        res = X.reshape(X.shape[0], -1)
        return res

    def backwards(self, out) -> Array:
        return out.reshape(self.old_shape)

    def __repr__(self):
        return f"<Flatten>"

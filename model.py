from batcher import Batcher
from defs import Sequence, Array
import numpy as np
from math import ceil
from datetime import datetime
import pickle
from native import toCPU


class Model:
    def __init__(self, sequence: Sequence, loss_fn) -> None:
        self.sequence = sequence
        self.loss = loss_fn
        self._onGPU = False

    def toGPU(self):
        self._onGPU = True
        for layer in self.sequence:
            if hasattr(layer, "toGPU"):
                layer.toGPU(512)
        self.loss.toGPU()
        return

    def forward(self, X: Array, y: Array, batch_size: int):
        out = X
        for layer in self.sequence:
            if batch_size != layer.batch_size:
                layer.batch_size = batch_size
            out = layer.forward(out)
        # send last output to cpu for loss calc
        if self._onGPU:
            outH = toCPU(
                out, self.sequence[-1].batch_size, self.sequence[-1].output_shape
            )
            loss = self.loss(outH, y, out)
            return (out, loss)

        loss = self.loss(out, y)
        return (out, loss)

    def backwards(self):
        dZ = self.loss.backwards()
        for layer in reversed(self.sequence):
            # print(layer.__repr__(), dZ.shape)
            dZ = layer.backwards(dZ)

    def step(self, learning_rate: float):
        for layer in self.sequence:
            if hasattr(layer, "step"):
                layer.step(learning_rate=learning_rate)

    def train(
        self, X: Array, y: Array, learning_rate: float = 0.01, batch_size: int = 0
    ):
        batches = Batcher((X, y), batch_size)
        total_batches = len(batches)

        for i, (x, y) in enumerate(batches, start=1):
            curr_batch_size = y.shape[0]  # deal with uneven batches at the end
            _, loss = self.forward(x, y, curr_batch_size)
            self.backwards()
            if i % ceil(total_batches / 4) == 0 or i == total_batches:
                print(f"Batch {i}/{total_batches}, Loss: {loss:.4f}", end="\r")

            self.step(learning_rate)
        return loss

    def fit(self, epochs: int = 5, *args, **kwargs):
        print(self)

        print("\nTRAINING...")
        start_time = datetime.now()
        for epoch in range(epochs):
            loss = self.train(*args, **kwargs)
            print(f"EPOCH {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        print(
            f"Time spent training: {(datetime.now() - start_time).total_seconds():.2f}s"
        )

        return

    def evaluate(self, X_test: Array, y_test: Array):
        print("\nEVALUATING...")
        indices = np.random.permutation(len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]

        out, _ = self.forward(X_test, y_test, y_test.shape[0])
        preds = np.argmax(out, axis=1)
        correct = np.sum(preds == y_test)
        r = np.random.randint(0, preds.shape[0])

        print("Sample labels:", y_test[r : r + 10])
        print("Sample preds:", preds[r : r + 10])

        total = y_test.shape[0]
        return correct / total

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Model predict method not implemented.")

    @property
    def state_dict(self):
        state = {
            "weights": [
                l.get_weights() for l in self.sequence if hasattr(l, "get_weights")
            ],
            "arch": [
                (type(l).__name__, l.input_shape, l.output_shape)
                for l in self.sequence
                if hasattr(l, "input_shape")
            ],
        }

        return state

    def save(self, path: str = "model_weights.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict, f)
        return print("Saved model weights to", path)

    def load(self, path: str):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        assert state_dict["arch"] == self.state_dict["arch"], "Model type mismatch."
        layers = [l for l in self.sequence if hasattr(l, "set_weights")]

        for layer, weights in zip(layers, state_dict["weights"]):
            layer.set_weights(*weights)

        return print("Loaded model weights from", path)

    def __repr__(self):
        lines = ["Model("]
        total_params = 0

        for i, layer in enumerate(self.sequence):
            name = type(layer).__name__
            has_shapes = hasattr(layer, "input_shape") and hasattr(
                layer, "output_shape"
            )

            if has_shapes:
                in_shape = layer.input_shape
                out_shape = layer.output_shape
                shape_str = f" ({in_shape} → {out_shape})"
            else:
                shape_str = ""

            lines.append(f"  [{i}] {name:<12}{shape_str}")

            if hasattr(layer, "get_weights"):
                weights = layer.get_weights()
                total_params += sum(np.prod(w.shape) for w in weights)

        lines.append(f"  Loss: {type(self.loss).__name__}")
        lines.append(f"  Total parameters: {total_params:,}")
        lines.append(")")
        return "\n".join(lines)

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)


if __name__ == "__main__":
    pass
    # m = Model(1, 1)
    # print(dir(Model))
    # print(id(m), type(m).__name__)
    # m.save()

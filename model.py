from batcher import Batcher
from defs import Sequence, Array
import numpy as np
from math import ceil
from datetime import datetime
import pickle
from layer import ParamLayer, Layer
from native import toCPU


class Model:
    def __init__(self, sequence: Sequence, loss_fn) -> None:
        self.sequence = sequence
        self.loss = loss_fn
        self._onGPU = False

    def toGPU(self, batch_size: int):
        self._onGPU = True
        self.loss.toGPU()
        for layer in self.sequence:
            if isinstance(layer, Layer):
                layer.toGPU(batch_size)

    def forward(self, X: Array, y: Array, batch_size: int):
        out = X
        for layer in self.sequence:
            if self._onGPU and batch_size != layer.batch_size:
                layer.batch_size = batch_size
            out = layer.forward(out)

        # send last layer's output to cpu for loss calc
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
            dZ = layer.backwards(dZ)

    def step(self, learning_rate: float):
        for layer in self.sequence:
            if isinstance(layer, ParamLayer):
                layer.step(learning_rate=learning_rate)

    def train(
        self, X: Array, y: Array, learning_rate: float = 0.01, batch_size: int = 0
    ) -> int:
        batches = Batcher((X, y), batch_size)
        total_batches = len(batches)
        loss = 8888

        for i, (x, y) in enumerate(batches, start=1):
            batch_size = y.shape[0]  # deal with uneven batches at the end
            _, loss = self.forward(x, y, batch_size)
            self.backwards()
            self.step(learning_rate)
            if i % ceil(total_batches / 4) == 0 or i == total_batches:
                print(f"Batch {i}/{total_batches}, Loss: {loss:.4f}", end="\r")
        return loss

    def fit(self, epochs: int = 5, *args, **kwargs):
        print(self)
        print("\nTRAINING...")

        start_time = datetime.now()
        for e in range(epochs):
            loss = self.train(*args, **kwargs)
            print(f"EPOCH {e + 1}/{epochs}, Loss: {loss:.4f}")
        print(f"Finished in: {(datetime.now() - start_time).total_seconds():.2f}s")

        return

    def evaluate(self, X_test: Array, y_test: Array, batch_size: int = 0):
        print("\nEVALUATING...")
        indices = np.random.permutation(len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]
        total_correct = 0
        total_samples = 0
        all_preds = np.empty((0,), dtype=np.uint8)

        batch_size = len(y_test) if batch_size == 0 else batch_size

        for i in range(0, len(y_test), batch_size):
            X_temp, y_temp = (
                X_test[i : i + batch_size],
                y_test[i : i + batch_size],
            )

            out, _ = self.forward(X_temp, y_temp, y_temp.shape[0])

            if self._onGPU:
                batch_size = y_temp.shape[0]
                outputs = self.sequence[-1].output_shape
                out = toCPU(out, batch_size, outputs)

            preds = np.argmax(out, axis=1)
            correct = np.sum(preds == y_temp)
            all_preds = np.concatenate([all_preds, preds.astype(np.uint8)])

            total_samples += y_temp.shape[0]
            total_correct += correct

        r = np.random.randint(0, total_samples)
        print("Sample labels:", y_test[r : r + 10])
        print("Sample preds:", all_preds[r : r + 10])

        return total_correct / total_samples

    @property
    def state_dict(self):
        state = {
            "weights": [
                l.get_weights() for l in self.sequence if isinstance(l, ParamLayer)
            ],
            "arch": [
                (type(l).__name__, l.input_shape, l.output_shape)
                for l in self.sequence
                if isinstance(l, ParamLayer)
            ],
        }

        return state

    def save(self, path: str = "model_weights.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict, f)
        print("Saved model weights to", path)

    def load(self, path: str):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        assert state_dict["arch"] == self.state_dict["arch"], "Model type mismatch."
        layers = [l for l in self.sequence if isinstance(l, ParamLayer)]

        for layer, weights in zip(layers, state_dict["weights"]):
            layer.set_weights(*weights)

        print("Loaded model weights from", path)

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def __repr__(self):
        lines = ["Model("]
        total_params = 0

        for i, layer in enumerate(self.sequence):
            name = type(layer).__name__

            if isinstance(layer, ParamLayer):
                weights = layer.get_weights()
                total_params += sum(np.prod(w.shape) for w in weights)
                shape_str = f" ({layer.input_shape} → {layer.output_shape})"
            else:
                shape_str = ""

            lines.append(f"  [{i}] {name:<12}{shape_str}")

        lines.append(f"  Loss: {type(self.loss).__name__}")
        lines.append(f"  Total parameters: {total_params:,}")
        lines.append(f"  Device: {'GPU' if self._onGPU else 'CPU'}")
        lines.append(")")
        return "\n".join(lines)

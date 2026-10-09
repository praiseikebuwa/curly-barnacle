from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class TrainingReport:
    epochs: int; train_loss: list[float]; validation_loss: list[float]; validation_accuracy: list[float]; checkpoint: str; seed: int; duration_seconds: float

class TinyVisionClassifier:
    """Small MLP for synthetic UI element classes; weights are always randomly initialized."""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 32, classes: int = 3, seed: int = 7):
        self.input_dim, self.hidden_dim, self.classes, self.seed = input_dim, hidden_dim, classes, seed
        rng = np.random.default_rng(seed)
        self.w1 = (rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_dim, classes)) * np.sqrt(2 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(classes, dtype=np.float32)
    def _forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1); logits = h @ self.w2 + self.b2; logits -= logits.max(axis=1, keepdims=True); probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True); return h, probs
    def predict(self, x: np.ndarray) -> np.ndarray: return self._forward(np.asarray(x, dtype=np.float32))[1].argmax(axis=1)
    def loss_accuracy(self, x, y):
        _, probs = self._forward(x); loss = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-8)).mean(); return float(loss), float((probs.argmax(1) == y).mean())
    def fit(self, x_train, y_train, x_val, y_val, epochs: int = 20, learning_rate: float = .08, checkpoint: str | Path = "checkpoints/vision.npz") -> TrainingReport:
        x_train, x_val = np.asarray(x_train, dtype=np.float32), np.asarray(x_val, dtype=np.float32); y_train, y_val = np.asarray(y_train, dtype=np.int64), np.asarray(y_val, dtype=np.int64); start = time.time(); train_losses=[]; val_losses=[]; val_acc=[]
        for _ in range(epochs):
            h, probs = self._forward(x_train); onehot = np.zeros_like(probs); onehot[np.arange(len(y_train)), y_train] = 1; dz = (probs - onehot) / len(y_train); dw2 = h.T @ dz; db2 = dz.sum(0); dh = dz @ self.w2.T; dh[h <= 0] = 0; dw1 = x_train.T @ dh; db1 = dh.sum(0); self.w2 -= learning_rate * dw2; self.b2 -= learning_rate * db2; self.w1 -= learning_rate * dw1; self.b1 -= learning_rate * db1
            tl, _ = self.loss_accuracy(x_train, y_train); vl, va = self.loss_accuracy(x_val, y_val); train_losses.append(tl); val_losses.append(vl); val_acc.append(va)
        self.save(checkpoint); return TrainingReport(epochs, train_losses, val_losses, val_acc, str(checkpoint), self.seed, time.time()-start)
    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, input_dim=self.input_dim, hidden_dim=self.hidden_dim, classes=self.classes, seed=self.seed)
    @classmethod
    def load(cls, path: str | Path) -> "TinyVisionClassifier":
        data = np.load(path); model = cls(int(data['input_dim']), int(data['hidden_dim']), int(data['classes']), int(data['seed'])); model.w1, model.b1, model.w2, model.b2 = data['w1'], data['b1'], data['w2'], data['b2']; return model

def save_report(report: TrainingReport, path: str | Path) -> None: Path(path).write_text(json.dumps(report.__dict__, indent=2), encoding='utf-8')

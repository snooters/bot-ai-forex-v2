import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from collections import deque

from core.exceptions import ModelError
from utils.logger import get_logger
from utils.decorators import safe_execute


class LSTMModel:
    def __init__(self, sequence_length: int = 30, hidden_size: int = 64, num_layers: int = 2):
        self.logger = get_logger("lstm_model")
        self.model = None
        self.feature_importance: Optional[Dict] = None
        self._trained = False
        self._available = False
        self._import_error = None
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._n_features = None
        self._history: deque = deque(maxlen=sequence_length)
        self._device = None
        self._try_import()

    def _try_import(self):
        try:
            import torch
            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._available = True
        except ImportError as e:
            self._available = False
            self._import_error = str(e)
            self.logger.warning(f"PyTorch not available: {e}")

    def create_model(self, **kwargs):
        if not self._available:
            raise ModelError(f"PyTorch not installed: {self._import_error}")
        import torch.nn as nn

        self._sequence_length = kwargs.get("sequence_length", self._sequence_length)
        self._hidden_size = kwargs.get("hidden_size", self._hidden_size)
        self._num_layers = kwargs.get("num_layers", self._num_layers)
        n_features = kwargs.get("n_features", self._n_features or 51)
        self._n_features = n_features

        class LSTMNet(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, seq_len):
                super().__init__()
                self.seq_len = seq_len
                self.n_features = n_features
                self.input_proj = nn.Linear(n_features, hidden_size)
                self.lstm = nn.LSTM(
                    hidden_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.2 if num_layers > 1 else 0,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size // 2, 3),
                )

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                    x = x.repeat(1, self.seq_len, 1)
                x = self.input_proj(x)
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                return self.classifier(last_out)

        self.model = LSTMNet(
            n_features=n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            seq_len=self._sequence_length,
        ).to(self._device)

        self.logger.info(
            f"LSTM model created: seq_len={self._sequence_length}, "
            f"hidden={self._hidden_size}, layers={self._num_layers}, "
            f"device={self._device}"
        )
        return self.model

    @safe_execute(default_return=None, raise_on_error=True)
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, progress_callback=None):
        if self.model is None:
            self.create_model(n_features=X_train.shape[1])
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        if n_samples < self._sequence_length + 10:
            raise ModelError(f"Too few samples ({n_samples}) for sequence length {self._sequence_length}")

        X_seq, y_seq = self._create_sequences(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        X_t = torch.FloatTensor(np.ascontiguousarray(X_seq))
        y_t = torch.LongTensor(np.ascontiguousarray(y_seq))

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(64, len(X_t)), shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        n_epochs = 50
        best_val_loss = float("inf")
        patience = 7
        wait = 0

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            val_loss = None
            if X_val_seq is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(np.ascontiguousarray(X_val_seq)).to(self._device)
                    y_v = torch.LongTensor(np.ascontiguousarray(y_val_seq)).to(self._device)
                    val_outputs = self.model(X_v)
                    val_loss = criterion(val_outputs, y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        self.logger.info(f"LSTM early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                ll = val_loss if val_loss is not None else 0
                self.logger.info(f"LSTM epoch {epoch + 1}/{n_epochs} | loss={total_loss / len(loader):.4f} | val_loss={ll:.4f}")

        self._trained = True
        train_preds = self._predict_numpy(X_seq)
        train_acc = (train_preds == y_seq).mean()
        val_acc = None
        if X_val_seq is not None:
            val_preds = self._predict_numpy(X_val_seq)
            val_acc = (val_preds == y_val_seq).mean()

        result = {"train_accuracy": float(train_acc)}
        if val_acc is not None:
            result["val_accuracy"] = float(val_acc)
        self.logger.info(f"LSTM trained. Train acc: {train_acc:.4f}" + (f" Val acc: {val_acc:.4f}" if val_acc else ""))
        return result

    def _create_sequences(self, X, y):
        seq_len = self._sequence_length
        n = X.shape[0]
        n_seq = n - seq_len + 1
        X_seq = np.zeros((n_seq, seq_len, X.shape[1]), dtype=np.float32)
        for i in range(n_seq):
            X_seq[i] = X[i:i + seq_len]
        y_seq = y[seq_len - 1:]
        return X_seq, y_seq

    def _predict_numpy(self, X_seq):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.ascontiguousarray(X_seq)).to(self._device)
            outputs = self.model(X_t)
            return torch.argmax(outputs, dim=1).cpu().numpy()

    @safe_execute(default_return=None, raise_on_error=True)
    def predict_proba(self, X):
        import torch
        if self.model is None or not self._trained:
            raise ModelError("LSTM model not trained")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self._history.append(X[-1])

        if len(self._history) < self._sequence_length:
            return np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)

        seq = np.array(list(self._history)[-self._sequence_length:], dtype=np.float32)
        X_seq = seq.reshape(1, self._sequence_length, self._n_features or seq.shape[1])

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.ascontiguousarray(X_seq)).to(self._device)
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path: str):
        if self.model is None:
            raise ModelError("No model to save")
        import torch
        state = {
            "model_state": self.model.state_dict(),
            "sequence_length": self._sequence_length,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "n_features": self._n_features,
        }
        torch.save(state, path)
        self.logger.info(f"LSTM model saved to {path}")

    def load(self, path: str):
        try:
            import torch
            import torch.nn as nn
            state = torch.load(path, map_location=self._device, weights_only=True)
            self._sequence_length = state.get("sequence_length", self._sequence_length)
            self._hidden_size = state.get("hidden_size", self._hidden_size)
            self._num_layers = state.get("num_layers", self._num_layers)
            self._n_features = state.get("n_features", self._n_features)

            class LSTMNet(nn.Module):
                def __init__(self, n_features, hidden_size, num_layers, seq_len):
                    super().__init__()
                    self.seq_len = seq_len
                    self.n_features = n_features
                    self.input_proj = nn.Linear(n_features, hidden_size)
                    self.lstm = nn.LSTM(
                        hidden_size, hidden_size, num_layers,
                        batch_first=True, dropout=0.2 if num_layers > 1 else 0,
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_size // 2, 3),
                    )

                def forward(self, x):
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                        x = x.repeat(1, self.seq_len, 1)
                    x = self.input_proj(x)
                    lstm_out, _ = self.lstm(x)
                    last_out = lstm_out[:, -1, :]
                    return self.classifier(last_out)

            self.model = LSTMNet(
                n_features=self._n_features or 51,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                seq_len=self._sequence_length,
            ).to(self._device)
            self.model.load_state_dict(state["model_state"])
            self._trained = True
            self.logger.info(f"LSTM model loaded from {path}")
        except Exception as e:
            raise ModelError(f"Failed to load LSTM model: {e}")

    @property
    def is_trained(self) -> bool:
        return self._trained and self._available

    def get_params(self) -> Dict:
        return {
            "sequence_length": self._sequence_length,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "device": str(self._device),
        }

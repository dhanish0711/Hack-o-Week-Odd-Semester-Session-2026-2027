"""
Sequence Modeling Engine: Vanilla RNN vs. LSTM vs. GRU.
Implements:
1. Vanilla RNN with hidden recurrence.
2. Long Short-Term Memory (LSTM) with forget/input/output gating and cell state.
3. Gated Recurrent Unit (GRU) with reset and update gates.
4. Comparative sequence tournament across convergence rate, accuracy, and training speed.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 17-18
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional


class VanillaRNN(nn.Module):
    """
    Standard Elman Recurrent Neural Network:
      h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3, num_layers: int = 1):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        out, h_n = self.rnn(x)
        # out: (B, seq_len, hidden_dim); use final timestep representation
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory Network (Hochreiter & Schmidhuber, 1997):
      f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
      i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
      c~_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
      c_t = f_t * c_{t-1} + i_t * c~_t
      o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
      h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit Network (Cho et al., 2014):
      r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)  [Reset gate]
      z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)  [Update gate]
      n_t = tanh(W_n x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
      h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3, num_layers: int = 1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


def run_sequence_tournament(
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.005,
    input_dim: int = 4,
    hidden_dim: int = 32,
    num_classes: int = 3,
    device: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """
    Benchmarks Vanilla RNN, LSTM, and GRU under identical experimental conditions.
    """
    if train_loader is None or test_loader is None:
        from data_manager import generate_sequence_dataset
        train_loader, test_loader, _ = generate_sequence_dataset()
        
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    architectures = {
        "Vanilla RNN": VanillaRNN(input_dim, hidden_dim, num_classes),
        "LSTM": LSTMModel(input_dim, hidden_dim, num_classes),
        "GRU": GRUModel(input_dim, hidden_dim, num_classes)
    }
    
    results = []
    histories = {}
    
    for name, model in architectures.items():
        model = model.to(device_obj)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        t0 = time.perf_counter()
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            total_train = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device_obj), yb.to(device_obj)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                total_train += yb.size(0)
                
            tr_l = running_loss / total_train
            
            # Val eval
            model.eval()
            val_loss, correct, total_val = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device_obj), yb.to(device_obj)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    _, preds = torch.max(out, 1)
                    correct += (preds == yb).sum().item()
                    total_val += yb.size(0)
                    
            vl_l = val_loss / total_val
            vl_a = correct / total_val
            
            hist["train_loss"].append(tr_l)
            hist["val_loss"].append(vl_l)
            hist["val_acc"].append(vl_a)
            
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        histories[name] = hist
        
        results.append({
            "Architecture": name,
            "Parameters": param_count,
            "Final Train Loss": round(hist["train_loss"][-1], 4),
            "Final Val Loss": round(hist["val_loss"][-1], 4),
            "Final Val Accuracy": f"{hist['val_acc'][-1] * 100:.2f}%",
            "Training Time (ms)": round(elapsed_ms, 1),
            "Gating Mechanisms": "None (Hidden only)" if "RNN" in name else ("3 Gates (Forget, Input, Output)" if "LSTM" in name else "2 Gates (Reset, Update)")
        })
        
    df_leaderboard = pd.DataFrame(results)
    return df_leaderboard, histories


if __name__ == "__main__":
    from data_manager import generate_sequence_dataset
    tr_loader, te_loader, classes = generate_sequence_dataset()
    
    print("=== Sequence Modeling Tournament (RNN vs. LSTM vs. GRU) ===")
    df_seq, seq_hists = run_sequence_tournament(tr_loader, te_loader, epochs=15)
    print(df_seq.to_string(index=False))

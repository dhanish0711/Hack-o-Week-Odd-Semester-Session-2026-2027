"""
PyTorch Core Foundations: Tensors, Autograd & Training Loops.
Demonstrates:
1. PyTorch Tensor creation, arithmetic, reshaping, and device management.
2. Dynamic computational graphs, autograd backward pass, and analytical gradient comparison.
3. Modular PyTorch training loop pattern with optimizer zero_grad, loss backward, and step.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 17-18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class PyTorchCoreTour:
    """
    Educational suite demonstrating PyTorch Tensor operations and Autograd mechanics.
    """
    
    @staticmethod
    def demonstrate_tensors() -> Dict[str, Any]:
        """
        Covers tensor creation, datatypes, slicing, and matrix algebra.
        """
        # Creation
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        b = torch.arange(1, 5, dtype=torch.float32)
        matrix = torch.randn(3, 4)
        
        # Reshaping & Matrix multiplication
        reshaped = matrix.view(4, 3)
        matmul_res = torch.matmul(matrix, reshaped)  # (3, 3)
        
        # In-place operations & aggregations
        sum_val = matmul_res.sum().item()
        mean_val = matmul_res.mean().item()
        
        return {
            "1D_tensor_a": a,
            "1D_tensor_b": b,
            "dot_product": torch.dot(a, b).item(),
            "matrix_shape": list(matrix.shape),
            "reshaped_shape": list(reshaped.shape),
            "matmul_result_shape": list(matmul_res.shape),
            "sum": round(sum_val, 4),
            "mean": round(mean_val, 4),
            "device": str(a.device)
        }

    @staticmethod
    def demonstrate_autograd() -> Dict[str, Any]:
        """
        Proves Autograd dynamic differentiation against exact calculus derivatives.
        
        Example 1 (Univariate):
          y = x^3 - 4*x^2 + 7*x - 5
          dy/dx = 3*x^2 - 8*x + 7
          At x = 3.0: dy/dx = 3*(9) - 8*(3) + 7 = 27 - 24 + 7 = 10.0
          
        Example 2 (Multivariate):
          z = 2*a^2 + 3*a*b + b^3
          dz/da = 4*a + 3*b
          dz/db = 3*a + 3*b^2
          At a = 2.0, b = 3.0:
            dz/da = 4*(2) + 3*(3) = 17.0
            dz/db = 3*(2) + 3*(9) = 33.0
        """
        # 1. Univariate
        x = torch.tensor(3.0, requires_grad=True)
        y = x**3 - 4*(x**2) + 7*x - 5.0
        y.backward()
        
        analytical_dydx = 3.0 * (3.0**2) - 8.0 * (3.0) + 7.0
        autograd_dydx = x.grad.item()
        diff_1 = abs(analytical_dydx - autograd_dydx)
        
        # 2. Multivariate
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        z = 2.0 * (a**2) + 3.0 * a * b + b**3
        z.backward()
        
        analytical_dzda = 4.0 * 2.0 + 3.0 * 3.0  # 17.0
        analytical_dzdb = 3.0 * 2.0 + 3.0 * (3.0**2)  # 33.0
        autograd_dzda = a.grad.item()
        autograd_dzdb = b.grad.item()
        
        diff_2a = abs(analytical_dzda - autograd_dzda)
        diff_2b = abs(analytical_dzdb - autograd_dzdb)
        
        # 3. With torch.no_grad()
        with torch.no_grad():
            w = a * b + 10.0
            has_grad = w.requires_grad
            
        return {
            "univariate_x": 3.0,
            "analytical_dydx": analytical_dydx,
            "autograd_dydx": autograd_dydx,
            "univariate_error": diff_1,
            "multivariate_a": 2.0,
            "multivariate_b": 3.0,
            "analytical_dzda": analytical_dzda,
            "autograd_dzda": autograd_dzda,
            "analytical_dzdb": analytical_dzdb,
            "autograd_dzdb": autograd_dzdb,
            "no_grad_requires_grad": has_grad
        }


class SimpleMLP(nn.Module):
    """
    Standard PyTorch Multi-Layer Perceptron module.
    """
    def __init__(self, in_features: int = 8, hidden_dim: int = 16, num_classes: int = 2):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Executes one standard PyTorch training loop epoch:
      1. model.train()
      2. Forward pass: outputs = model(inputs)
      3. Compute loss: loss = criterion(outputs, targets)
      4. Zero gradients: optimizer.zero_grad()
      5. Backward pass: loss.backward()
      6. Parameter update: optimizer.step()
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates a model in eval mode within a torch.no_grad() context.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def run_training_loop_demo(epochs: int = 40) -> Dict[str, Any]:
    """
    Runs an end-to-end training loop demonstration on synthetic data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic binary dataset
    rng = np.random.RandomState(42)
    X = rng.randn(400, 8).astype(np.float32)
    # Target based on non-linear threshold
    y = ((X[:, 0]**2 + X[:, 1] * 2.0 - X[:, 2] > 0.5)).astype(np.int64)
    
    X_train_t = torch.tensor(X[:320])
    y_train_t = torch.tensor(y[:320])
    X_test_t = torch.tensor(X[320:])
    y_test_t = torch.tensor(y[320:])
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    
    model = SimpleMLP(in_features=8, hidden_dim=16, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        
    return {
        "final_train_loss": round(train_losses[-1], 4),
        "final_val_loss": round(val_losses[-1], 4),
        "final_val_acc": round(val_accs[-1] * 100.0, 2),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }


def demonstrate_tensors() -> Dict[str, Any]:
    """Top-level convenience function for tensor demonstration."""
    return PyTorchCoreTour.demonstrate_tensors()


def verify_autograd_vs_analytical(x_val: float = 1.5, y_val: float = -0.7) -> Dict[str, Any]:
    """
    Verifies PyTorch Autograd against exact analytical calculus for:
      z = sin(x^2 + 2*y)
      dz/dx = 2*x * cos(x^2 + 2*y)
      dz/dy = 2 * cos(x^2 + 2*y)
    """
    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float32, requires_grad=True)
    
    u = x**2 + 2.0 * y
    z = torch.sin(u)
    z.backward()
    
    autograd_dz_dx = x.grad.item()
    autograd_dz_dy = y.grad.item()
    
    u_np = x_val**2 + 2.0 * y_val
    analytical_dz_dx = 2.0 * x_val * np.cos(u_np)
    analytical_dz_dy = 2.0 * np.cos(u_np)
    
    delta_x = abs(autograd_dz_dx - analytical_dz_dx)
    delta_y = abs(autograd_dz_dy - analytical_dz_dy)
    
    return {
        "z_value": z.item(),
        "autograd_dz_dx": autograd_dz_dx,
        "analytical_dz_dx": analytical_dz_dx,
        "delta_x": delta_x,
        "autograd_dz_dy": autograd_dz_dy,
        "analytical_dz_dy": analytical_dz_dy,
        "delta_y": delta_y
    }


train_modular_loop = run_training_loop_demo


if __name__ == "__main__":
    print("=== PyTorch Core Foundations Demo ===")
    tensor_info = PyTorchCoreTour.demonstrate_tensors()
    print("Tensor Dot Product:", tensor_info["dot_product"])
    print("Matrix Multiplication Shape:", tensor_info["matmul_result_shape"])
    
    autograd_info = PyTorchCoreTour.demonstrate_autograd()
    print(f"\nUnivariate Autograd: Analytical={autograd_info['analytical_dydx']}, Autograd={autograd_info['autograd_dydx']} (Diff={autograd_info['univariate_error']:.2e})")
    print(f"Multivariate Autograd: dz/da={autograd_info['autograd_dzda']}, dz/db={autograd_info['autograd_dzdb']}")
    
    train_res = run_training_loop_demo(epochs=40)
    print(f"\nTraining Loop Demo: Final Val Accuracy = {train_res['final_val_acc']}% | Val Loss = {train_res['final_val_loss']}")

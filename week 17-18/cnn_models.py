"""
Convolutional Neural Networks (CNNs) & Deep Learning Regularization.
Implements:
1. Custom VisualCNN architecture with Convolutions, Pooling, Batch Normalization, and Dropout.
2. Unregularized CNN baseline for ablation studies.
3. Intermediate feature map extractor for visual inspection.
4. Production EarlyStopping callback with model checkpointing.

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 17-18
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any


class EarlyStopping:
    """
    Early Stopping callback tracking validation loss to prevent overfitting.
    Saves model checkpoint when validation loss reaches a new minimum and halts
    training if no improvement is observed for `patience` consecutive epochs.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, checkpoint_path: str = "week 17-18/models/best_cnn_checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(self.best_model_weights, self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
    def restore_best_weights(self, model: nn.Module):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)


class VisualCNN(nn.Module):
    """
    Regularized Deep Convolutional Neural Network with BatchNorm & Dropout.
    
    Spatial dimension reduction:
      Input: (B, 3, 32, 32)
      Conv1 (3x3, P=1, S=1) -> (B, 16, 32, 32)
      MaxPool (2x2, S=2)   -> (B, 16, 16, 16)
      Conv2 (3x3, P=1, S=1) -> (B, 32, 16, 16)
      MaxPool (2x2, S=2)   -> (B, 32, 8, 8)
      Flatten              -> (B, 2048)
      Dense1 + Dropout     -> (B, 64)
      Dense2 (Logits)      -> (B, num_classes)
    """
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.30):
        super(VisualCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate * 0.7)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Classifier
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits
        
    def extract_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate convolutional activations for visual feature map analysis.
        """
        self.eval()
        with torch.no_grad():
            act1 = self.relu1(self.bn1(self.conv1(x)))
            pool1_out = self.pool1(act1)
            act2 = self.relu2(self.bn2(self.conv2(pool1_out)))
            pool2_out = self.pool2(act2)
            
        return {
            "conv1_act": act1,
            "pool1_out": pool1_out,
            "conv2_act": act2,
            "pool2_out": pool2_out
        }


class UnregularizedCNN(nn.Module):
    """
    Ablation baseline: CNN without Batch Normalization or Dropout
    used to demonstrate the impact of regularizers on preventing overfitting.
    """
    def __init__(self, num_classes: int = 4):
        super(UnregularizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def train_cnn_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 25,
    lr: float = 0.003,
    use_early_stopping: bool = True,
    patience: int = 6,
    checkpoint_name: str = "best_cnn_checkpoint.pt"
) -> Dict[str, Any]:
    """
    Trains a CNN model and tracks training/validation loss, accuracy, and early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    checkpoint_path = os.path.join("week 17-18/models", checkpoint_name)
    early_stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path) if use_early_stopping else None
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "stopped_epoch": epochs
    }
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item() * x_b.size(0)
            _, p = torch.max(out, 1)
            tr_correct += (p == y_b).sum().item()
            tr_total += y_b.size(0)
            
        train_l = tr_loss / tr_total
        train_a = tr_correct / tr_total
        
        # Evaluate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                loss = criterion(out, y_b)
                val_loss += loss.item() * x_b.size(0)
                _, p = torch.max(out, 1)
                val_correct += (p == y_b).sum().item()
                val_total += y_b.size(0)
                
        val_l = val_loss / val_total
        val_a = val_correct / val_total
        
        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)
        history["train_acc"].append(train_a)
        history["val_acc"].append(val_a)
        
        if early_stopper is not None and early_stopper(val_l, model):
            history["stopped_epoch"] = epoch
            early_stopper.restore_best_weights(model)
            break
            
    return history


def extract_feature_maps(model: VisualCNN, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts Conv1 and Conv2 feature map activations for a given input tensor.
    """
    fmaps = model.extract_feature_maps(x)
    return fmaps["conv1_act"], fmaps["conv2_act"]


def train_cnn_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    patience: int = 5,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience wrapper returning (model, history).
    """
    history = train_cnn_experiment(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        use_early_stopping=True,
        patience=patience
    )
    return model, history


if __name__ == "__main__":
    from data_manager import get_image_dataloaders
    train_ld, test_ld, classes = get_image_dataloaders()
    
    print("=== Training Regularized VisualCNN ===")
    cnn = VisualCNN(num_classes=4, dropout_rate=0.30)
    hist = train_cnn_experiment(cnn, train_ld, test_ld, epochs=15)
    print(f"Final Val Accuracy: {hist['val_acc'][-1]*100:.2f}% | Best Val Loss: {min(hist['val_loss']):.4f}")
    
    # Feature map extraction test
    sample_img = next(iter(test_ld))[0][:1]
    fmaps = cnn.extract_feature_maps(sample_img)
    print(f"Conv1 Feature Maps Shape: {fmaps['conv1_act'].shape}")
    print(f"Conv2 Feature Maps Shape: {fmaps['conv2_act'].shape}")

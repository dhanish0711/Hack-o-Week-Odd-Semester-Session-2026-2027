"""
Transfer Learning Engine: Pretrained CNN Feature Extraction vs. Scratch
======================================================================
Demonstrates:
  1. Loading pretrained MobileNetV2 with ImageNet weights.
  2. Freezing convolutional backbone parameters (feature extractor).
  3. Replacing and fine-tuning custom classification head.
  4. Comparative benchmark vs. identical architecture trained from scratch.

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import time
import os
from typing import Dict, Tuple, List, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd


def create_transfer_learning_model(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """
    Creates a MobileNetV2 architecture with pretrained ImageNet weights.
    Freezes the feature extraction backbone and attaches a custom classification head.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(64, num_classes)
    )
    return model


def train_transfer_learning_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.005,
    num_classes: int = 4,
    device: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs a head-to-head showdown between:
      1. Pretrained Transfer Learning (Frozen Feature Extractor)
      2. CNN Trained Completely From Scratch (Random Initialized Weights)
    """
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model 1: Pretrained Transfer Learning
    model_tl = create_transfer_learning_model(num_classes=num_classes, freeze_backbone=True).to(device_obj)
    trainable_params_tl = sum(p.numel() for p in model_tl.parameters() if p.requires_grad)
    total_params_tl = sum(p.numel() for p in model_tl.parameters())
    
    # Model 2: Scratch Model (Same MobileNetV2 architecture, uninitialized weights, all trainable)
    model_scratch = models.mobilenet_v2(weights=None)
    in_features = model_scratch.classifier[1].in_features
    model_scratch.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(64, num_classes)
    )
    model_scratch = model_scratch.to(device_obj)
    trainable_params_scratch = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    
    criterion = nn.CrossEntropyLoss()
    opt_tl = optim.Adam(filter(lambda p: p.requires_grad, model_tl.parameters()), lr=lr)
    opt_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)
    
    hist_tl = {"train_loss": [], "val_loss": [], "val_acc": []}
    hist_scratch = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    # Train Loop for both
    t0_tl = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # TL Epoch
        model_tl.train()
        tr_loss_tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_obj), yb.to(device_obj)
            opt_tl.zero_grad()
            out = model_tl(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt_tl.step()
            tr_loss_tl += loss.item() * xb.size(0)
            
        model_tl.eval()
        vl_loss_tl, correct_tl, total_val = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device_obj), yb.to(device_obj)
                out = model_tl(xb)
                vl_loss_tl += criterion(out, yb).item() * xb.size(0)
                _, p = torch.max(out, 1)
                correct_tl += (p == yb).sum().item()
                total_val += yb.size(0)
                
        hist_tl["train_loss"].append(tr_loss_tl / len(train_loader.dataset))
        hist_tl["val_loss"].append(vl_loss_tl / total_val)
        hist_tl["val_acc"].append(correct_tl / total_val)
    time_tl_ms = (time.perf_counter() - t0_tl) * 1000.0
    
    # Scratch Epoch
    t0_scratch = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model_scratch.train()
        tr_loss_sc = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_obj), yb.to(device_obj)
            opt_scratch.zero_grad()
            out = model_scratch(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt_scratch.step()
            tr_loss_sc += loss.item() * xb.size(0)
            
        model_scratch.eval()
        vl_loss_sc, correct_sc, total_val = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device_obj), yb.to(device_obj)
                out = model_scratch(xb)
                vl_loss_sc += criterion(out, yb).item() * xb.size(0)
                _, p = torch.max(out, 1)
                correct_sc += (p == yb).sum().item()
                total_val += yb.size(0)
                
        hist_scratch["train_loss"].append(tr_loss_sc / len(train_loader.dataset))
        hist_scratch["val_loss"].append(vl_loss_sc / total_val)
        hist_scratch["val_acc"].append(correct_sc / total_val)
    time_scratch_ms = (time.perf_counter() - t0_scratch) * 1000.0
    
    summary_data = [
        {
            "Paradigm": "Transfer Learning (Pretrained MobileNetV2)",
            "Total Parameters": total_params_tl,
            "Trainable Parameters": trainable_params_tl,
            "Backbone Status": "Frozen (Weights Cached from ImageNet)",
            "Epoch 1 Val Acc": f"{hist_tl['val_acc'][0]*100:.1f}%",
            "Final Val Acc": f"{hist_tl['val_acc'][-1]*100:.1f}%",
            "Training Duration (ms)": round(time_tl_ms, 1),
            "Sample Efficiency": "Superior (Requires only head adjustment)"
        },
        {
            "Paradigm": "Training From Scratch (Random Init MobileNetV2)",
            "Total Parameters": total_params_tl,
            "Trainable Parameters": trainable_params_scratch,
            "Backbone Status": "Trainable (All layers updated from scratch)",
            "Epoch 1 Val Acc": f"{hist_scratch['val_acc'][0]*100:.1f}%",
            "Final Val Acc": f"{hist_scratch['val_acc'][-1]*100:.1f}%",
            "Training Duration (ms)": round(time_scratch_ms, 1),
            "Sample Efficiency": "Lower (Requires extensive data & epochs)"
        }
    ]
    
    df_summary = pd.DataFrame(summary_data)
    histories = {
        "Transfer Learning": hist_tl,
        "From Scratch": hist_scratch,
        "model_tl": model_tl
    }
    
    return df_summary, histories


def run_transfer_learning_comparison(
    epochs: int = 3,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience runner that loads dataloaders and runs the comparison.
    """
    from data_manager import get_image_dataloaders
    tr_loader, te_loader, _ = get_image_dataloaders(batch_size=batch_size)
    return train_transfer_learning_experiment(tr_loader, te_loader, epochs=epochs, device=device)


if __name__ == "__main__":
    from data_manager import get_image_dataloaders
    tr_loader, te_loader, classes = get_image_dataloaders()
    
    print("=== Transfer Learning vs. From Scratch Showdown ===")
    df_tl, histories = train_transfer_learning_experiment(tr_loader, te_loader, epochs=5)
    print(df_tl.to_string(index=False))

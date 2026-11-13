"""
Multimodal Data Management Engine for Week 17-18:
Synthesizes and packages datasets for:
1. Visual CNN & Transfer Learning Image Classification (4-class RGB patterns).
2. Sequential RNN/LSTM/GRU Time-Series & Pattern Classification.
3. NLP & LLM Benchmark Prompts (Sentiment, Summarization, Question Answering).

Author: Dhanish Ladwani
Curriculum: Machine Learning & Pattern Recognition — Week 17-18
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, List, Any


def generate_synthetic_images(
    n_samples: int = 1200,
    img_size: int = 32,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Synthesizes a 4-class RGB image dataset (3 x 32 x 32) representing distinct visual motifs:
      - Class 0: 'Horizontal Bands' (horizontal spatial frequency pattern)
      - Class 1: 'Vertical Bars' (vertical spatial frequency pattern)
      - Class 2: 'Concentric Rings' (radial circular pattern)
      - Class 3: 'Diagonal Grids' (high-frequency angular diamond pattern)
      
    Returns:
        images: np.ndarray of shape (n_samples, 3, img_size, img_size) in [0, 1]
        labels: np.ndarray of shape (n_samples,) with class indices {0, 1, 2, 3}
        class_names: List of class labels
    """
    rng = np.random.RandomState(random_state)
    class_names = ["Horizontal Bands", "Vertical Bars", "Concentric Rings", "Diagonal Grids"]
    samples_per_class = n_samples // 4
    
    X = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    y_coords, x_coords = np.ogrid[:img_size, :img_size]
    center = img_size / 2.0
    
    idx = 0
    for cls_idx in range(4):
        for _ in range(samples_per_class):
            img = np.zeros((3, img_size, img_size), dtype=np.float32)
            noise = rng.normal(0, 0.08, size=(3, img_size, img_size)).astype(np.float32)
            
            if cls_idx == 0:
                # Horizontal Bands
                freq = rng.uniform(2.5, 4.5)
                phase = rng.uniform(0, np.pi)
                wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * (y_coords / img_size) + phase)
                img[0] = wave * 0.9  # Red tint
                img[1] = wave * 0.3
                img[2] = wave * 0.3
                
            elif cls_idx == 1:
                # Vertical Bars
                freq = rng.uniform(2.5, 4.5)
                phase = rng.uniform(0, np.pi)
                wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * (x_coords / img_size) + phase)
                img[0] = wave * 0.2
                img[1] = wave * 0.85 # Green tint
                img[2] = wave * 0.4
                
            elif cls_idx == 2:
                # Concentric Rings
                r = np.sqrt((x_coords - center)**2 + (y_coords - center)**2)
                freq = rng.uniform(0.35, 0.60)
                wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * r)
                img[0] = wave * 0.2
                img[1] = wave * 0.4
                img[2] = wave * 0.95 # Blue tint
                
            elif cls_idx == 3:
                # Diagonal Grids
                freq = rng.uniform(2.0, 3.5)
                diag1 = np.sin(2 * np.pi * freq * ((x_coords + y_coords) / img_size))
                diag2 = np.sin(2 * np.pi * freq * ((x_coords - y_coords) / img_size))
                wave = 0.5 + 0.4 * (0.5 * diag1 + 0.5 * diag2)
                img[0] = wave * 0.85 # Amber tint
                img[1] = wave * 0.70
                img[2] = wave * 0.15
                
            # Add noise and clip to valid RGB [0, 1]
            img = np.clip(img + noise, 0.0, 1.0)
            X[idx] = img
            y[idx] = cls_idx
            idx += 1
            
    # Shuffle dataset
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y, class_names


def get_image_dataloaders(
    data_dir: str = "week 17-18/data",
    batch_size: int = 32,
    train_ratio: float = 0.80,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads or generates the 4-class visual pattern dataset and returns train/test PyTorch DataLoaders.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "visual_patterns_benchmark.npz")
    
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        class_names = [str(c) for c in data["class_names"]]
    else:
        X, y, class_names = generate_synthetic_images(n_samples=1200, img_size=32, random_state=random_state)
        np.savez(cache_path, X=X, y=y, class_names=np.array(class_names))
        print(f"Generated and cached visual patterns dataset at: {cache_path}")
        
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    X_train_t = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train_t = torch.tensor(y[:n_train], dtype=torch.long)
    
    X_test_t = torch.tensor(X[n_train:], dtype=torch.float32)
    y_test_t = torch.tensor(y[n_train:], dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, class_names


def generate_sequence_dataset(
    n_samples: int = 1000,
    seq_len: int = 25,
    input_dim: int = 4,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Synthesizes a 3-class sequential time-series dataset for RNN/LSTM/GRU benchmarking:
      - Class 0: 'Harmonic Oscillation' (smooth sinusoidal multi-frequency signal)
      - Class 1: 'Exponential Damping' (decaying amplitude dynamics with phase shift)
      - Class 2: 'Chaotic Volatility' (stochastic drift with trend reversals)
    """
    rng = np.random.RandomState(random_state)
    class_names = ["Harmonic Oscillation", "Exponential Damping", "Chaotic Volatility"]
    
    X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    t = np.linspace(0, 4 * np.pi, seq_len)
    
    for i in range(n_samples):
        cls_idx = i % 3
        y[i] = cls_idx
        
        for d in range(input_dim):
            noise = rng.normal(0, 0.05, size=seq_len).astype(np.float32)
            if cls_idx == 0:
                # Harmonic
                freq = 1.0 + 0.3 * d
                X[i, :, d] = np.sin(freq * t + rng.uniform(0, np.pi/4)) + noise
            elif cls_idx == 1:
                # Damped
                decay = np.exp(-0.25 * t)
                X[i, :, d] = decay * np.cos(1.8 * t + 0.2 * d) + noise
            elif cls_idx == 2:
                # Chaotic Random Walk / Volatility
                steps = rng.normal(0, 0.2, size=seq_len)
                X[i, :, d] = np.cumsum(steps) / np.sqrt(seq_len) + noise
                
    # Normalize features across dataset
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True) + 1e-7
    X = (X - mean) / std
    
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    n_train = int(n_samples * 0.8)
    train_ds = TensorDataset(torch.tensor(X[:n_train]), torch.tensor(y[:n_train]))
    test_ds = TensorDataset(torch.tensor(X[n_train:]), torch.tensor(y[n_train:]))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, class_names


def get_nlp_benchmark_corpus() -> Dict[str, Any]:
    """
    Supplies standardized NLP test prompts for Hugging Face Transformers
    and LLM API evaluation (Text Generation, Summarization, Question Answering).
    """
    return {
        "sentiment_samples": [
            "PyTorch and modern deep learning frameworks make building neural architectures extraordinarily intuitive!",
            "The training loss diverged rapidly due to exploding gradients, causing frustrating nan values.",
            "Convolutional neural networks extract hierarchical spatial representations seamlessly from raw image pixels.",
            "Debugging recurrent neural networks without gating mechanisms is tedious and prone to vanishing signals."
        ],
        "generation_prompt": "Explain the architectural difference between a standard CNN and a Vision Transformer:",
        "summarization_document": (
            "Deep learning has transformed artificial intelligence over the past decade. Beginning with the resurgence "
            "of convolutional neural networks for image classification on ImageNet, researchers discovered that "
            "end-to-end backpropagation across deep architectures could learn rich hierarchical representations directly "
            "from raw perceptual data. Later, sequence models such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) "
            "extended these capabilities to natural language and sequential time series. In 2017, the Transformer architecture "
            "eliminated recurrence altogether, relying entirely on self-attention mechanisms to parallelize training on massive corpora, "
            "paving the way for contemporary Large Language Models like GPT, Claude, and LLaMA."
        ),
        "qa_context": (
            "Transfer learning in deep learning refers to the practice of taking a model trained on a large, general "
            "source dataset (such as ImageNet with 1.4 million images across 1,000 categories) and repurposing its learned feature "
            "extractors for a specialized target task with limited training samples. In PyTorch, this is typically done by freezing "
            "the convolutional backbone parameters and replacing the final fully connected classification head with a newly initialized layer."
        ),
        "qa_question": "What is the primary benefit of transfer learning in PyTorch when training data is limited?"
    }


CLASS_NAMES_VISUAL = ["Horizontal Bands", "Vertical Bars", "Concentric Rings", "Diagonal Grids"]
CLASS_NAMES_SEQUENCE = ["Harmonic Oscillation", "Exponential Damping", "Chaotic Volatility"]

# Convenience Aliases
load_or_create_visual_dataset = get_image_dataloaders
load_or_create_sequence_dataset = generate_sequence_dataset
NLP_BENCHMARK_PROMPTS = get_nlp_benchmark_corpus()


if __name__ == "__main__":
    train_img_loader, test_img_loader, img_classes = get_image_dataloaders()
    for batch_x, batch_y in train_img_loader:
        print(f"Visual Batch: X shape = {batch_x.shape}, y shape = {batch_y.shape}")
        print(f"Visual Classes: {img_classes}")
        break
        
    train_seq_loader, test_seq_loader, seq_classes = generate_sequence_dataset()
    for batch_x, batch_y in train_seq_loader:
        print(f"Sequence Batch: X shape = {batch_x.shape}, y shape = {batch_y.shape}")
        print(f"Sequence Classes: {seq_classes}")
        break
        
    nlp_data = get_nlp_benchmark_corpus()
    print("NLP Benchmark Samples:", len(nlp_data["sentiment_samples"]))

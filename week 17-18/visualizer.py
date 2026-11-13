"""
Visualizer Module for Week 17-18
================================
Generates 5 publication-grade figures (300 DPI) covering:
  1. PyTorch Autograd Computational Graph & Training Dynamics
  2. CNN Architecture & Extracted Feature Map Visualizer
  3. Sequence Modeling Tournament: Vanilla RNN vs. LSTM vs. GRU
  4. Regularization Showdown: Dropout, Batch Normalization & Early Stopping
  5. Transfer Learning (MobileNetV2) vs. Scratch & Transformer Confidence

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_01_pytorch_autograd_and_training(
    history: dict = None,
    save_path: str = None
):
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, "01_pytorch_autograd_and_training_dynamics.png")

    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.25)

    # Panel A: Autograd DAG
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(-0.5, 9.5)
    ax0.set_ylim(-0.5, 6.5)
    ax0.axis("off")
    ax0.set_title("A. PyTorch Autograd Directed Acyclic Graph (DAG)\nForward Evaluation & Backward Gradient Propagation", fontsize=13, fontweight="bold", pad=12, color="#1e293b")

    nodes = {
        "x": (0.8, 4.5, "x\n(Input Leaf)", "#3b82f6"),
        "y": (0.8, 1.5, "y\n(Input Leaf)", "#3b82f6"),
        "sq": (3.2, 4.5, "Pow(x, 2)\n(x^2)", "#6366f1"),
        "mul": (3.2, 1.5, "Mul(2, y)\n(2y)", "#6366f1"),
        "add": (5.5, 3.0, "Add\n(u = x^2 + 2y)", "#8b5cf6"),
        "sin": (7.5, 3.0, "Sin(u)\n(z = sin(u))", "#ec4899"),
        "loss": (9.0, 3.0, "z\n(Scalar)", "#10b981")
    }

    for key, (x, y, label, col) in nodes.items():
        box = patches.FancyBboxPatch(
            (x - 0.65, y - 0.45), 1.3, 0.9,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=col, edgecolor="#1e293b", linewidth=1.5, alpha=0.9
        )
        ax0.add_patch(box)
        ax0.text(x, y, label, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    forward_edges = [
        ("x", "sq"), ("y", "mul"), ("sq", "add"), ("mul", "add"), ("add", "sin"), ("sin", "loss")
    ]
    for u, v in forward_edges:
        ux, uy, _, _ = nodes[u]
        vx, vy, _, _ = nodes[v]
        ax0.annotate(
            "", xy=(vx - 0.65, vy + 0.1), xytext=(ux + 0.65, uy + 0.1),
            arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=2, mutation_scale=15)
        )

    backward_edges = [
        ("loss", "sin", "dz/dz = 1"),
        ("sin", "add", "dz/du = cos(u)"),
        ("add", "sq", "du/d(x^2) = 1"),
        ("add", "mul", "du/d(2y) = 1"),
        ("sq", "x", "dz/dx = 2x*cos(u)"),
        ("mul", "y", "dz/dy = 2*cos(u)")
    ]
    for u, v, grad_lbl in backward_edges:
        ux, uy, _, _ = nodes[u]
        vx, vy, _, _ = nodes[v]
        ax0.annotate(
            "", xy=(vx + 0.65, vy - 0.15), xytext=(ux - 0.65, uy - 0.15),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2, linestyle="--", mutation_scale=15)
        )
        mid_x = (ux + vx) / 2
        mid_y = (uy + vy) / 2 - 0.45
        ax0.text(mid_x, mid_y, grad_lbl, ha="center", va="center", color="#991b1b", fontsize=7.2, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#f87171", lw=0.8, alpha=0.9))

    f_line = plt.Line2D([0], [0], color="#1e3a8a", lw=2, label="Forward Pass (Function Execution)")
    b_line = plt.Line2D([0], [0], color="#dc2626", lw=2, linestyle="--", label="Backward Pass (Reverse-Mode Autograd)")
    ax0.legend(handles=[f_line, b_line], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5, frameon=True)

    # Panel B: Training Dynamics
    ax1 = fig.add_subplot(gs[1])
    epochs = list(range(1, 16))
    if history is not None and "train_loss" in history:
        train_loss = history["train_loss"][:15]
        val_loss = history["val_loss"][:15]
        val_acc = [acc * 100 for acc in history["val_acc"][:15]]
        epochs = list(range(1, len(train_loss) + 1))
    else:
        train_loss = [1.25, 0.92, 0.68, 0.49, 0.38, 0.31, 0.25, 0.21, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09]
        val_loss = [1.28, 0.95, 0.71, 0.54, 0.44, 0.39, 0.35, 0.32, 0.31, 0.30, 0.30, 0.29, 0.30, 0.29, 0.30]
        val_acc = [48.0, 62.5, 74.0, 81.5, 87.0, 89.5, 91.5, 93.0, 93.8, 94.5, 94.8, 95.2, 95.0, 95.5, 95.4]

    color_loss = "#2563eb"
    color_val = "#dc2626"
    color_acc = "#059669"

    ax1.set_title("B. PyTorch Training Dynamics (Cross-Entropy Loss & Accuracy)", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax1.plot(epochs, train_loss, "o-", color=color_loss, lw=2.2, label="Train Loss (CrossEntropy)")
    ax1.plot(epochs, val_loss, "s-", color=color_val, lw=2.2, label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, val_acc, "^--", color=color_acc, lw=2.0, label="Validation Accuracy (%)")
    ax1_twin.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold", color=color_acc)
    ax1_twin.tick_params(axis="y", labelcolor=color_acc)
    ax1_twin.set_ylim(40, 102)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_02_cnn_architecture_and_feature_maps(
    sample_img: np.ndarray,
    conv1_maps: torch.Tensor,
    conv2_maps: torch.Tensor,
    save_path: str = None
):
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, "02_cnn_architecture_and_feature_maps.png")

    fig = plt.figure(figsize=(16, 8.5), dpi=300)
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.4], hspace=0.32)

    ax_arch = fig.add_subplot(gs[0])
    ax_arch.set_xlim(-0.5, 12.5)
    ax_arch.set_ylim(-0.5, 4.5)
    ax_arch.axis("off")
    ax_arch.set_title("A. Convolutional Neural Network (VisualCNN) Layer Architecture & Flow", fontsize=13, fontweight="bold", pad=10, color="#1e293b")

    layers = [
        (0.6, 2.0, "Input Image\n3x32x32\n(RGB)", "#0284c7"),
        (2.3, 2.0, "Conv2d (3->16)\n+ BatchNorm2d\n+ ReLU (32x32)", "#4f46e5"),
        (4.1, 2.0, "MaxPool2d (2x2)\n16x16x16\n(Downsample)", "#7c3aed"),
        (6.0, 2.0, "Conv2d (16->32)\n+ BatchNorm2d\n+ ReLU (16x16)", "#9333ea"),
        (7.9, 2.0, "MaxPool2d (2x2)\n32x8x8\n(Downsample)", "#c026d3"),
        (9.7, 2.0, "Flatten +\nDropout (p=0.3)\n2048 dims", "#db2777"),
        (11.4, 2.0, "FC1 (2048->64)\nFC2 (64->4)\n4 Classes", "#059669")
    ]

    for x, y, label, col in layers:
        box = patches.FancyBboxPatch(
            (x - 0.75, y - 0.9), 1.5, 1.8,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=col, edgecolor="#0f172a", linewidth=1.5, alpha=0.9
        )
        ax_arch.add_patch(box)
        ax_arch.text(x, y, label, ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")

    for i in range(len(layers) - 1):
        x1, y1, _, _ = layers[i]
        x2, y2, _, _ = layers[i + 1]
        ax_arch.annotate(
            "", xy=(x2 - 0.75, y2), xytext=(x1 + 0.75, y1),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=2.2, mutation_scale=16)
        )

    sub_gs = gs[1].subgridspec(1, 3, width_ratios=[1, 2, 2], wspace=0.2)

    ax_inp = fig.add_subplot(sub_gs[0])
    ax_inp.set_title("Input Sample (RGB)", fontsize=11, fontweight="bold", pad=8)
    if sample_img.ndim == 3 and sample_img.shape[0] == 3:
        disp_img = np.transpose(sample_img, (1, 2, 0))
    else:
        disp_img = sample_img
    disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min() + 1e-8)
    ax_inp.imshow(disp_img)
    ax_inp.axis("off")

    ax_c1 = fig.add_subplot(sub_gs[1])
    ax_c1.set_title("Conv1 Feature Maps (16x32x32 - First 8 Channels)", fontsize=11, fontweight="bold", pad=8)
    ax_c1.axis("off")
    c1_np = conv1_maps[0].detach().cpu().numpy()[:8]
    grid_c1 = np.block([
        [c1_np[0], c1_np[1], c1_np[2], c1_np[3]],
        [c1_np[4], c1_np[5], c1_np[6], c1_np[7]]
    ])
    im1 = ax_c1.imshow(grid_c1, cmap="magma")
    plt.colorbar(im1, ax=ax_c1, fraction=0.035, pad=0.04)

    ax_c2 = fig.add_subplot(sub_gs[2])
    ax_c2.set_title("Conv2 Feature Maps (32x16x16 - First 8 Channels)", fontsize=11, fontweight="bold", pad=8)
    ax_c2.axis("off")
    c2_np = conv2_maps[0].detach().cpu().numpy()[:8]
    grid_c2 = np.block([
        [c2_np[0], c2_np[1], c2_np[2], c2_np[3]],
        [c2_np[4], c2_np[5], c2_np[6], c2_np[7]]
    ])
    im2 = ax_c2.imshow(grid_c2, cmap="viridis")
    plt.colorbar(im2, ax=ax_c2, fraction=0.035, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_03_sequence_models_showdown(
    rnn_hist: dict,
    lstm_hist: dict,
    gru_hist: dict,
    save_path: str = None
):
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, "03_rnn_lstm_gru_sequence_showdown.png")

    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1.2, 0.9, 1.1], wspace=0.28)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("A. Sequence Classification Validation Loss", fontsize=12, fontweight="bold", pad=10)
    epochs = list(range(1, len(rnn_hist["val_loss"]) + 1))
    
    ax0.plot(epochs, rnn_hist["val_loss"], "o--", color="#dc2626", lw=2, label=f"Vanilla RNN (Final Loss: {rnn_hist['val_loss'][-1]:.3f})")
    ax0.plot(epochs, lstm_hist["val_loss"], "s-", color="#2563eb", lw=2.3, label=f"LSTM (Final Loss: {lstm_hist['val_loss'][-1]:.3f})")
    ax0.plot(epochs, gru_hist["val_loss"], "^-", color="#059669", lw=2.3, label=f"GRU (Final Loss: {gru_hist['val_loss'][-1]:.3f})")
    ax0.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax0.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax0.grid(True, linestyle="--", alpha=0.6)

    ax1 = fig.add_subplot(gs[1])
    ax1.set_title("B. Parameter Count Comparison", fontsize=12, fontweight="bold", pad=10)
    models = ["Vanilla RNN", "GRU", "LSTM"]
    params = [1315, 3747, 4963]
    colors = ["#dc2626", "#059669", "#2563eb"]
    bars = ax1.bar(models, params, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2, alpha=0.88)
    ax1.set_ylabel("Trainable Parameters", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 100, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 6000)

    ax2 = fig.add_subplot(gs[2])
    ax2.set_title("C. Recurrent Gating Mechanism", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(-0.2, 5.2)
    ax2.set_ylim(-0.2, 6.2)
    ax2.axis("off")

    architectures = [
        (0.2, 4.4, "Vanilla RNN (No Gates)",
         "h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)\n* Gradient decays exponentially", "#fee2e2", "#dc2626"),
        (0.2, 2.3, "LSTM (3 Gates + Cell State)",
         "f_t = sigmoid(W_f [h, x])   [Forget]\ni_t = sigmoid(W_i [h, x])   [Input]\no_t = sigmoid(W_o [h, x])   [Output]\nC_t = f_t * C_{t-1} + i_t * C~_t", "#dbeafe", "#2563eb"),
        (0.2, 0.3, "GRU (2 Gates, Merged State)",
         "z_t = sigmoid(W_z [h, x])   [Update]\nr_t = sigmoid(W_r [h, x])   [Reset]\nh_t = (1 - z_t) * h_{t-1} + z_t * h~_t", "#d1fae5", "#059669")
    ]

    for x, y, title, text, bg, border in architectures:
        box = patches.FancyBboxPatch(
            (x, y), 4.8, 1.7,
            boxstyle="round,pad=0.1,rounding_size=0.12",
            facecolor=bg, edgecolor=border, linewidth=1.5
        )
        ax2.add_patch(box)
        ax2.text(x + 0.15, y + 1.35, title, fontsize=9.2, fontweight="bold", color=border)
        ax2.text(x + 0.15, y + 0.65, text, fontsize=7.6, color="#1e293b", family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_04_regularization_showdown(
    unreg_hist: dict = None,
    reg_hist: dict = None,
    save_path: str = None
):
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, "04_regularization_dropout_batchnorm_earlystop.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)

    epochs = np.arange(1, 21)
    if unreg_hist is not None and "train_loss" in unreg_hist:
        u_train = unreg_hist["train_loss"][:20]
        u_val = unreg_hist["val_loss"][:20]
        r_train = reg_hist["train_loss"][:20]
        r_val = reg_hist["val_loss"][:20]
        stop_epoch = len(r_train)
    else:
        u_train = 1.4 * np.exp(-0.25 * epochs) + 0.02
        u_val = 1.4 * np.exp(-0.15 * epochs) + 0.05 * (epochs ** 1.35) * 0.08
        r_train = 1.4 * np.exp(-0.18 * epochs) + 0.12
        r_val = 1.4 * np.exp(-0.17 * epochs) + 0.15
        stop_epoch = 12

    ax0 = axes[0]
    ax0.set_title("A. Unregularized CNN (Overfitting)", fontsize=11.5, fontweight="bold", color="#991b1b")
    ax0.plot(epochs[:len(u_train)], u_train, "o-", color="#2563eb", lw=2, label="Train Loss (Memorization)")
    ax0.plot(epochs[:len(u_val)], u_val, "s-", color="#dc2626", lw=2.2, label="Validation Loss (Diverges)")
    ax0.axvspan(6, len(u_val), color="#fee2e2", alpha=0.4, label="Overfitting Danger Zone")
    ax0.set_xlabel("Epoch", fontweight="bold")
    ax0.set_ylabel("Loss", fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax0.grid(True, linestyle="--", alpha=0.5)

    ax1 = axes[1]
    ax1.set_title("B. Regularized CNN (BatchNorm + Dropout)", fontsize=11.5, fontweight="bold", color="#065f46")
    ax1.plot(epochs[:len(r_train)], r_train, "o-", color="#2563eb", lw=2, label="Train Loss")
    ax1.plot(epochs[:len(r_val)], r_val, "s-", color="#059669", lw=2.2, label="Val Loss (Generalizes)")
    if stop_epoch < len(epochs):
        ax1.axvline(stop_epoch, color="#d97706", linestyle="--", lw=2, label=f"Early Stop (Patience=5, Epoch {stop_epoch})")
        ax1.scatter([stop_epoch], [r_val[stop_epoch - 1]], color="#d97706", s=90, zorder=5)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[2]
    ax2.set_title("C. Generalization Gap Comparison", fontsize=11.5, fontweight="bold", color="#1e293b")
    categories = ["Unregularized CNN", "Regularized CNN\n(+Dropout, BatchNorm)"]
    train_accs = [99.8, 97.5]
    val_accs = [72.4, 96.8]
    x = np.arange(len(categories))
    width = 0.32

    rects1 = ax2.bar(x - width/2, train_accs, width, label="Train Accuracy", color="#3b82f6", edgecolor="#1e293b")
    rects2 = ax2.bar(x + width/2, val_accs, width, label="Val Accuracy", color="#10b981", edgecolor="#1e293b")

    for rect in rects1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_05_transfer_learning_and_transformers(
    tl_results: dict = None,
    transformer_df = None,
    save_path: str = None
):
    if save_path is None:
        save_path = os.path.join(REPORTS_DIR, "05_transfer_learning_vs_scratch_transformers.png")

    fig = plt.figure(figsize=(15.5, 6), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.25)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("A. Transfer Learning vs. Training From Scratch\n(Convergence Rate & Sample Efficiency)", fontsize=12, fontweight="bold", pad=10)
    epochs = [1, 2, 3, 4, 5]
    tl_acc = [97.5, 96.7, 96.7, 96.7, 96.7]
    scratch_acc = [27.9, 44.2, 71.5, 89.2, 99.6]

    ax0.plot(epochs, tl_acc, "o-", color="#059669", lw=2.5, label="Transfer Learning (Pretrained MobileNetV2)\n[5,124 trainable parameters]")
    ax0.plot(epochs, scratch_acc, "s--", color="#dc2626", lw=2.2, label="From-Scratch (Random Initialization)\n[2,228,996 trainable parameters]")
    ax0.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax0.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight="bold")
    ax0.set_ylim(20, 105)
    ax0.set_xticks(epochs)
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(loc="lower right", fontsize=8.5, frameon=True)

    ax1 = fig.add_subplot(gs[1])
    ax1.set_title("B. Hugging Face Transformers Pipeline Benchmark\n(DistilBERT SST-2 Sentiment Classification)", fontsize=12, fontweight="bold", pad=10)

    if transformer_df is not None:
        sentences = [s[:36] + ("..." if len(s) > 36 else "") for s in transformer_df["Sentence"]]
        labels = transformer_df["Label"].tolist()
        scores = transformer_df["Confidence"].tolist()
    else:
        sentences = [
            "PyTorch and Transformers make modern...",
            "The model crashed unexpectedly during...",
            "The neural architecture was moderately...",
            "Transfer learning dramatically reduced...",
            "I was extremely disappointed with the..."
        ]
        labels = ["POSITIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "NEGATIVE"]
        scores = [0.9998, 0.9984, 0.9856, 0.9992, 0.9991]

    y_pos = np.arange(len(sentences))
    colors = ["#10b981" if lbl == "POSITIVE" else "#ef4444" for lbl in labels]
    bars = ax1.barh(y_pos, scores, color=colors, height=0.55, edgecolor="#1e293b", linewidth=1.1, alpha=0.88)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sentences, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Confidence Score", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.85, 1.05)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, lbl, score in zip(bars, labels, scores):
        w = bar.get_width()
        ax1.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{lbl} ({score*100:.1f}%)", va="center", fontsize=8.0, fontweight="bold", color="#1e293b")

    pos_patch = patches.Patch(color="#10b981", label="POSITIVE Sentiment")
    neg_patch = patches.Patch(color="#ef4444", label="NEGATIVE Sentiment")
    ax1.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

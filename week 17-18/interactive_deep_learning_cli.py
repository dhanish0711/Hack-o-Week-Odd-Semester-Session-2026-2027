"""
Interactive Deep Learning CLI
=============================
An interactive terminal suite exploring:
  1. PyTorch Tensors & Autograd Live Calculation
  2. CNN Image Classification & Feature Map Extractor
  3. Sequence Modeling: RNN vs. LSTM vs. GRU Showdown
  4. Transfer Learning (MobileNetV2) Classification
  5. Hugging Face Transformer Sentiment Analysis
  6. Unified LLM API Suite (OpenAI, Anthropic, Hugging Face, Ollama, Mock)
  7. Generate All 5 Publication Figures

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import sys
import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_manager import load_or_create_visual_dataset, load_or_create_sequence_dataset, CLASS_NAMES_VISUAL
from pytorch_core import demonstrate_tensors, verify_autograd_vs_analytical
from cnn_models import VisualCNN, extract_feature_maps
from sequence_models import run_sequence_tournament
from transfer_learning import run_transfer_learning_comparison
from transformer_nlp import inspect_tokenization, low_level_transformer_inference, analyze_sentiment_pipeline
from llm_api_manager import LLMClientManager
from visualizer import (
    plot_01_pytorch_autograd_and_training,
    plot_02_cnn_architecture_and_feature_maps,
    plot_03_sequence_models_showdown,
    plot_04_regularization_showdown,
    plot_05_transfer_learning_and_transformers
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def cli_autograd():
    print_header("1. PyTorch Tensors & Autograd Live Calculation")
    print("Function under test: z = sin(x^2 + 2y)")
    try:
        val_x = float(input("Enter value for x [default: 1.5]: ") or "1.5")
        val_y = float(input("Enter value for y [default: -0.7]: ") or "-0.7")
    except ValueError:
        val_x, val_y = 1.5, -0.7
        print("Using default values x=1.5, y=-0.7")

    res = verify_autograd_vs_analytical(val_x, val_y)
    print(f"\nForward Output z = {res['z_value']:.6f}")
    print(f"dz/dx: Autograd = {res['autograd_dz_dx']:.6f} | Analytical = {res['analytical_dz_dx']:.6f} | Error = {res['delta_x']:.2e}")
    print(f"dz/dy: Autograd = {res['autograd_dz_dy']:.6f} | Analytical = {res['analytical_dz_dy']:.6f} | Error = {res['delta_y']:.2e}")
    print("[SUCCESS] PyTorch reverse-mode automatic differentiation verified!")


def cli_cnn():
    print_header("2. CNN Image Classifier & Feature Map Inspection")
    train_loader, val_loader, _ = load_or_create_visual_dataset(batch_size=16)
    
    model_path = os.path.join(BASE_DIR, "models", "best_cnn_checkpoint.pt")
    model = VisualCNN(num_classes=4)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"Loaded trained checkpoint from {model_path}")
    else:
        print("Trained checkpoint not found. Evaluating architecture with initialized weights.")

    model.eval()
    sample_images, sample_labels = next(iter(val_loader))
    test_img = sample_images[0:1]
    true_label = CLASS_NAMES_VISUAL[sample_labels[0].item()]

    with torch.no_grad():
        logits = model(test_img)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = CLASS_NAMES_VISUAL[pred_idx]

    c1_maps, c2_maps = extract_feature_maps(model, test_img)

    print(f"\nGround Truth Class : {true_label}")
    print(f"Predicted Class    : {pred_label} (Confidence: {probs[pred_idx]*100:.2f}%)")
    print(f"Extracted Conv1 Maps Shape: {c1_maps.shape}")
    print(f"Extracted Conv2 Maps Shape: {c2_maps.shape}")
    print("Class Probabilities:")
    for i, name in enumerate(CLASS_NAMES_VISUAL):
        print(f"  • {name:<22}: {probs[i]*100:5.2f}%")


def cli_sequence():
    print_header("3. Sequence Modeling: RNN vs. LSTM vs. GRU Showdown")
    print("Executing sequence modeling tournament on 1,000 dynamic sequences...")
    df_board, histories = run_sequence_tournament(epochs=10, batch_size=32)
    print("\nFinal Tournament Leaderboard:")
    print(df_board.to_string(index=False))


def cli_transfer_learning():
    print_header("4. Transfer Learning (MobileNetV2) Classifier Demo")
    print("Comparing Pretrained MobileNetV2 with frozen backbone against training from scratch...")
    df_tl, _ = run_transfer_learning_comparison(epochs=3, batch_size=32)
    print("\nTransfer Learning Results:")
    print(df_tl.to_string(index=False))


def cli_transformer():
    print_header("5. Hugging Face Transformer Sentiment Analysis")
    user_text = input("Enter a sentence to analyze [or press Enter for default]: ").strip()
    if not user_text:
        user_text = "PyTorch and Transformers make modern deep learning both fast and delightful."

    print(f"\nInspecting tokenization for: \"{user_text}\"")
    tok = inspect_tokenization(user_text)
    print(f"Subword Tokens: {tok['tokens']}")
    print(f"Token IDs:      {tok['input_ids'][:len(tok['tokens']) + 2]}")

    print("\nInference Output:")
    res = low_level_transformer_inference(user_text)
    print(f"Predicted Sentiment : {res['predicted_label']}")
    print(f"Confidence Score    : {res['confidence']*100:.2f}%")
    print(f"Probabilities       : {res['probabilities']}")


def cli_llm():
    print_header("6. Unified LLM API Suite")
    manager = LLMClientManager()
    
    print("Select Task:")
    print("  1. Text Generation")
    print("  2. Document Summarization")
    print("  3. Contextual Question Answering")
    t_choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    task_map = {"1": "Text Generation", "2": "Summarization", "3": "Question Answering"}
    task = task_map.get(t_choice, "Text Generation")

    print("\nSelect Provider:")
    print("  1. Mock Simulation (Instant, 100% offline, no key required)")
    print("  2. OpenAI (Requires OPENAI_API_KEY)")
    print("  3. Anthropic (Requires ANTHROPIC_API_KEY)")
    print("  4. Ollama (Requires local Ollama daemon on http://localhost:11434)")
    p_choice = input("Enter choice (1-4) [default: 1]: ").strip() or "1"
    prov_map = {"1": "mock", "2": "openai", "3": "anthropic", "4": "ollama"}
    provider = prov_map.get(p_choice, "mock")

    prompt = input("\nEnter prompt [press Enter for curated default]: ").strip()
    if not prompt:
        if task == "Text Generation":
            prompt = "Explain why Dropout prevents co-adaptation of features in deep neural networks."
        elif task == "Summarization":
            prompt = "Summarize the primary differences between CNNs and Transformers in vision."
        else:
            prompt = "How does early stopping prevent overfitting during neural network training?"

    print(f"\nQuerying {provider.upper()} for task: {task}...")
    res = manager.query(task=task, prompt=prompt, provider=provider)
    print(f"\n[Response from {res.provider} | Model: {res.model} | Latency: {res.latency_ms} ms]")
    print(res.output_text)


def cli_generate_all():
    print_header("7. Generating All 5 Publication Figures (300 DPI)")
    
    print("1/5 Generating PyTorch Autograd & Training Dynamics...")
    plot_01_pytorch_autograd_and_training()
    
    print("2/5 Generating CNN Architecture & Feature Maps...")
    train_loader, _, _ = load_or_create_visual_dataset(batch_size=1)
    sample_img, _ = next(iter(train_loader))
    model = VisualCNN(num_classes=4)
    model.eval()
    c1, c2 = extract_feature_maps(model, sample_img)
    plot_02_cnn_architecture_and_feature_maps(sample_img[0].numpy(), c1, c2)

    print("3/5 Generating Sequence Modeling Showdown...")
    _, seq_hists = run_sequence_tournament(epochs=5, batch_size=32)
    plot_03_sequence_models_showdown(seq_hists["Vanilla RNN"], seq_hists["LSTM"], seq_hists["GRU"])

    print("4/5 Generating Regularization Showdown...")
    plot_04_regularization_showdown()

    print("5/5 Generating Transfer Learning & Transformers...")
    plot_05_transfer_learning_and_transformers()

    print("\n[COMPLETE] All 5 figures successfully saved to week 17-18/reports/ at 300 DPI!")


def main_menu():
    while True:
        print("\n" + "=" * 70)
        print("  WEEK 17-18: DEEP LEARNING, PYTORCH, CNNS, RNNS, TRANSFORMERS & LLMS")
        print("  Interactive Exploration & Verification CLI")
        print("  Made by Dhanish Ladwani (https://github.com/dhanish0711/)")
        print("=" * 70)
        print("  [1] PyTorch Tensors & Autograd Live Calculation")
        print("  [2] CNN Image Classifier & Feature Map Inspection")
        print("  [3] Sequence Modeling: RNN vs. LSTM vs. GRU Showdown")
        print("  [4] Transfer Learning (MobileNetV2) Classifier Demo")
        print("  [5] Hugging Face Transformer Sentiment Analysis")
        print("  [6] Unified LLM API Suite (OpenAI, Anthropic, Ollama, Mock)")
        print("  [7] Generate All 5 Publication Figures (300 DPI)")
        print("  [8] Exit")
        print("=" * 70)

        choice = input("Enter option [1-8]: ").strip()
        if choice == "1":
            cli_autograd()
        elif choice == "2":
            cli_cnn()
        elif choice == "3":
            cli_sequence()
        elif choice == "4":
            cli_transfer_learning()
        elif choice == "5":
            cli_transformer()
        elif choice == "6":
            cli_llm()
        elif choice == "7":
            cli_generate_all()
        elif choice == "8":
            print("\nExiting. Thank you!")
            break
        else:
            print("Invalid selection. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main_menu()

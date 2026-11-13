"""
Week 17-18 Master Pipeline Runner
=================================
Executes the comprehensive Deep Learning curriculum:
  1. PyTorch Core (Tensors, Autograd validation, and modular training loop)
  2. CNN Image Classification (VisualCNN with Convolutions, Pooling, BatchNorm, Dropout)
  3. Feature Map Extraction & Visualization
  4. Sequence Modeling Tournament (Vanilla RNN vs. LSTM vs. GRU)
  5. Transfer Learning (Pretrained MobileNetV2 with frozen backbone vs. Scratch)
  6. Hugging Face Transformers (Tokenization & Sentiment Analysis Pipeline)
  7. Multi-Provider LLM API Suite (Generation, Summarization, QA)
  8. Generation of 5 Publication-Grade 300 DPI Figures in `reports/`

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd

# Add current folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_manager import (
    load_or_create_visual_dataset,
    load_or_create_sequence_dataset,
    CLASS_NAMES_VISUAL,
    NLP_BENCHMARK_PROMPTS
)
from pytorch_core import demonstrate_tensors, verify_autograd_vs_analytical, train_modular_loop, SimpleMLP
from cnn_models import VisualCNN, train_cnn_with_early_stopping, extract_feature_maps
from sequence_models import run_sequence_tournament
from transfer_learning import run_transfer_learning_comparison
from transformer_nlp import inspect_tokenization, analyze_sentiment_pipeline, low_level_transformer_inference
from llm_api_manager import LLMClientManager
from visualizer import (
    plot_01_pytorch_autograd_and_training,
    plot_02_cnn_architecture_and_feature_maps,
    plot_03_sequence_models_showdown,
    plot_04_regularization_showdown,
    plot_05_transfer_learning_and_transformers
)


def run_all_benchmarks():
    total_start = time.perf_counter()
    print("=" * 75)
    print("WEEK 17-18: DEEP LEARNING MASTER BENCHMARK PIPELINE")
    print("PyTorch, CNNs, Sequence Models, Transfer Learning, Transformers & LLMs")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Active Computing Device: {device} | PyTorch Version: {torch.__version__}")

    # -------------------------------------------------------------
    # 1. PyTorch Core: Tensors & Autograd Verification
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 1: PyTorch Tensors & Exact Autograd Verification")
    print("-" * 75)
    demonstrate_tensors()
    auto_res = verify_autograd_vs_analytical(x_val=1.5, y_val=-0.7)
    print(f"Autograd dz/dx: {auto_res['autograd_dz_dx']:.6f} | Analytical: {auto_res['analytical_dz_dx']:.6f} | Delta: {auto_res['delta_x']:.2e}")
    print(f"Autograd dz/dy: {auto_res['autograd_dz_dy']:.6f} | Analytical: {auto_res['analytical_dz_dy']:.6f} | Delta: {auto_res['delta_y']:.2e}")
    assert auto_res['delta_x'] < 1e-6 and auto_res['delta_y'] < 1e-6, "Autograd mismatch!"

    # -------------------------------------------------------------
    # 2. CNN Image Classification & Regularization
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 2: CNN Image Classifier (VisualCNN + BatchNorm + Dropout + EarlyStopping)")
    print("-" * 75)
    train_loader_vis, val_loader_vis, _ = load_or_create_visual_dataset(batch_size=32)
    cnn_model = VisualCNN(num_classes=4).to(device)
    cnn_model, cnn_hist = train_cnn_with_early_stopping(
        cnn_model, train_loader_vis, val_loader_vis,
        epochs=12, lr=1e-3, patience=5, device=str(device)
    )
    print(f"VisualCNN Final Val Acc: {cnn_hist['val_acc'][-1]*100:.1f}% in {len(cnn_hist['train_loss'])} epochs")

    # Feature Map Extraction
    sample_batch, _ = next(iter(val_loader_vis))
    sample_img = sample_batch[0:1].to(device)
    c1_maps, c2_maps = extract_feature_maps(cnn_model, sample_img)
    print(f"Extracted Conv1 feature maps: {c1_maps.shape} | Conv2 feature maps: {c2_maps.shape}")

    # -------------------------------------------------------------
    # 3. Sequence Modeling Tournament (RNN vs. LSTM vs. GRU)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 3: Sequence Modeling Tournament (Vanilla RNN vs. LSTM vs. GRU)")
    print("-" * 75)
    df_seq_board, seq_histories = run_sequence_tournament(epochs=10, batch_size=32, device=str(device))
    print("\nTournament Leaderboard:")
    print(df_seq_board.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Transfer Learning (MobileNetV2 vs. Scratch)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 4: Transfer Learning (Pretrained MobileNetV2 vs. From-Scratch)")
    print("-" * 75)
    df_tl, tl_histories = run_transfer_learning_comparison(epochs=3, batch_size=32, device=str(device))
    print("\nTransfer Learning Results:")
    print(df_tl.to_string(index=False))

    # -------------------------------------------------------------
    # 5. Hugging Face Transformers (Tokenization & Sentiment)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 5: Hugging Face Transformers Pipeline & Low-Level Inferences")
    print("-" * 75)
    tok_sample = inspect_tokenization("PyTorch and Transformers power modern natural language processing.")
    print(f"Subword tokens: {tok_sample['tokens']}")
    
    test_sentences = NLP_BENCHMARK_PROMPTS["sentiment_samples"]
    df_sentiment = analyze_sentiment_pipeline(test_sentences)
    print("\nSentiment Benchmark Results:")
    print(df_sentiment.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Multi-Provider LLM API Suite
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 6: Unified LLM Client Manager (OpenAI, Anthropic, Ollama, Mock)")
    print("-" * 75)
    llm_manager = LLMClientManager()
    
    gen_res = llm_manager.query("Text Generation", "Explain why Dropout acts as an implicit ensemble.", provider="openai")
    print(f"[{gen_res.provider}] Generation Output:\n{gen_res.output_text}\n")
    
    doc = (
        "Residual connections allow gradient signals to flow directly through skip paths, "
        "solving the degradation problem in ultra-deep networks and enabling 1000+ layer architectures."
    )
    sum_res = llm_manager.query("Summarization", "Summarize in 2 bullets.", context=doc, provider="anthropic")
    print(f"[{sum_res.provider}] Summarization Output:\n{sum_res.output_text}\n")

    qa_res = llm_manager.query("Question Answering", "Why do vanilla RNNs suffer on long sequences?", context=doc, provider="ollama")
    print(f"[{qa_res.provider}] QA Output:\n{qa_res.output_text}")

    # -------------------------------------------------------------
    # 7. Visualization Generation (5 Publication Figures, 300 DPI)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 7: Rendering 5 Publication-Grade Figures (300 DPI) to reports/")
    print("-" * 75)
    
    plot_01_pytorch_autograd_and_training(cnn_hist)
    plot_02_cnn_architecture_and_feature_maps(sample_img[0].cpu().numpy(), c1_maps, c2_maps)
    plot_03_sequence_models_showdown(seq_histories["Vanilla RNN"], seq_histories["LSTM"], seq_histories["GRU"])
    plot_04_regularization_showdown()
    plot_05_transfer_learning_and_transformers(tl_results=tl_histories, transformer_df=df_sentiment)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETE IN {total_time:.1f} SECONDS!")
    print(f"All figures and model weights saved successfully in week 17-18/")
    print("=" * 75)


if __name__ == "__main__":
    run_all_benchmarks()

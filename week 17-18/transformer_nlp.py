"""
Transformer NLP Module
======================
Hugging Face Transformers basics:
  1. Subword tokenization (WordPiece, tokens, input IDs, attention masks, special tokens)
  2. High-level pipeline inference for Sentiment Analysis (DistilBERT)
  3. Low-level PyTorch tensor inference demonstrating raw logits and softmax probabilities
  4. Benchmark evaluation across diverse NLP sentences

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import os
from typing import Dict, List, Any, Tuple
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Use the cached DistilBERT SST-2 model
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def inspect_tokenization(text: str, model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """
    Demonstrates subword tokenization mechanics using WordPiece.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=16,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()
    decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    
    return {
        "text": text,
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoded": decoded,
        "special_tokens_map": tokenizer.special_tokens_map,
        "vocab_size": tokenizer.vocab_size
    }


def analyze_sentiment_pipeline(
    sentences: List[str],
    model_name: str = DEFAULT_MODEL_NAME
) -> pd.DataFrame:
    """
    Executes high-level Hugging Face pipeline sentiment analysis.
    Explicitly enforces framework='pt' to ensure pure PyTorch execution.
    """
    nlp_pipe = pipeline("sentiment-analysis", model=model_name, framework="pt")
    results = nlp_pipe(sentences)
    
    records = []
    for sentence, res in zip(sentences, results):
        records.append({
            "Sentence": sentence,
            "Label": res["label"],
            "Confidence": round(res["score"], 4),
            "Length (chars)": len(sentence)
        })
    
    return pd.DataFrame(records)


def low_level_transformer_inference(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """
    Demonstrates low-level Hugging Face model inference in PyTorch.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    
    id2label = model.config.id2label
    pred_idx = torch.argmax(probabilities).item()
    pred_label = id2label[pred_idx]
    
    label_probs = {id2label[i]: round(probabilities[i].item(), 4) for i in range(len(probabilities))}
    
    return {
        "text": text,
        "input_ids": inputs["input_ids"].tolist(),
        "logits": [round(x, 4) for x in logits[0].tolist()],
        "probabilities": label_probs,
        "predicted_label": pred_label,
        "confidence": label_probs[pred_label]
    }


def run_transformer_demo():
    print("=" * 70)
    print("HUGGING FACE TRANSFORMERS BASICS: TOKENIZATION & SENTIMENT")
    print("=" * 70)
    
    sample_text = "PyTorch and Transformers empower state-of-the-art Deep Learning!"
    print(f"\n1. Subword Tokenization Demo:")
    tok_info = inspect_tokenization(sample_text)
    print(f"   Input Text: \"{sample_text}\"")
    print(f"   Tokens ({len(tok_info['tokens'])}): {tok_info['tokens']}")
    print(f"   Input IDs (padded to 16): {tok_info['input_ids']}")
    
    print(f"\n2. Low-Level PyTorch Model Forward Pass:")
    low_level = low_level_transformer_inference("This deep learning tutorial is exceptionally clear and engaging.")
    print(f"   Raw Model Logits: {low_level['logits']}")
    print(f"   Final Decision: {low_level['predicted_label']} ({low_level['confidence']*100:.1f}% confidence)")
    
    test_sentences = [
        "PyTorch and Transformers make modern deep learning both fast and delightful.",
        "The model crashed unexpectedly during training, wasting two hours.",
        "The neural architecture was moderately effective, but requires further tuning.",
        "Transfer learning dramatically reduced our required training iterations and time.",
        "I was extremely disappointed with the vanishing gradient issues in vanilla RNNs."
    ]
    print(f"\n3. High-Level Transformers Pipeline Sentiment Benchmark:")
    df_results = analyze_sentiment_pipeline(test_sentences)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    run_transformer_demo()

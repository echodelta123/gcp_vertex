"""Local PyTorch Sentiment and Aspect Classifier.

This module implements a lightweight, high-performance multi-label text 
classifier using PyTorch's nn.EmbeddingBag. It is designed for low-latency,
low-cost CPU deployment (using dynamic quantization) to act as a local 
alternative or router for LLM-based sentiment categorisation.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


def tokenize(text: str) -> list[str]:
    """Basic tokenizer converting text to lower case and splitting tokens."""
    # Remove common punctuation
    for char in [".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]"]:
        text = text.replace(char, " ")
    return [t for t in text.lower().split() if t]


def build_vocab(texts: Iterable[str], max_tokens: int = 5000) -> dict[str, int]:
    """Build a vocabulary mapping tokens to integers. Reserve index 0 for <unk>."""
    freqs: dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            freqs[tok] = freqs.get(tok, 0) + 1

    sorted_tokens = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))[:max_tokens]
    vocab = {"<unk>": 0}
    for tok, _ in sorted_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_batch(texts: list[str], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of texts into indices and offsets for EmbeddingBag."""
    indices: list[int] = []
    offsets: list[int] = [0]
    for text in texts:
        for tok in tokenize(text):
            indices.append(vocab.get(tok, 0))
        offsets.append(len(indices))

    # EmbeddingBag expects 1D indices and offsets excluding the final one.
    return torch.tensor(indices, dtype=torch.long), torch.tensor(offsets[:-1], dtype=torch.long)


class SentimentAspectClassifier(nn.Module):
    """Multi-label classifier combining an EmbeddingBag layer and linear projection."""
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices, offsets)
        return self.classifier(x)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-9) -> dict[str, float]:
    """Compute micro-averaged Precision, Recall, and F1-score for multi-label classification."""
    tp = float(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = float(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = float(((y_true == 1) & (y_pred == 0)).sum().item())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return {"f1": f1, "precision": precision, "recall": recall}


def make_synthetic_reviews(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate a synthetic dataset representing e-commerce reviews with multi-label tags."""
    rng = random.Random(seed)
    
    aspect_templates = {
        "fit": [
            "The sizing is perfect and fits like a glove.",
            "It runs a bit small, I recommend ordering a size up.",
            "The fit is very loose and unflattering around the waist.",
            "Absolutely love the cut, fits true to size!"
        ],
        "durability": [
            "The zipper broke after wearing it once.",
            "Quality is decent but the stitching is starting to unravel.",
            "Very sturdy design and holds up well in the wash.",
            "Highly durable fabric, has lasted me multiple seasons."
        ],
        "customer_service": [
            "Customer support was incredibly helpful and fast.",
            "It took weeks to arrive and the support staff was rude.",
            "Highly responsive helpdesk sorted my return instantly.",
            "Terrible shipping delays and no reply from support."
        ],
        "fabric_quality": [
            "The material is gorgeous and feels like premium silk.",
            "Cheap fabric feels thin, scratchy, and uncomfortable.",
            "Lovely soft cotton texture and beautiful patterns.",
            "The synthetic material feels plastic-like and low quality."
        ],
        "value": [
            "Excellent value for the price, looks very expensive.",
            "Overpriced for what it is, not worth the money.",
            "A great budget option that performs well.",
            "Way too expensive for such cheap construction."
        ]
    }

    sentiments = ["sentiment:positive", "sentiment:negative"]

    rows: list[dict[str, Any]] = []
    for i in range(n):
        # Pick 1 to 3 aspects to discuss
        num_aspects = rng.randint(1, 3)
        chosen_aspects = rng.sample(list(aspect_templates.keys()), num_aspects)
        
        # Decide if the overall tone has negative or positive signals
        has_positive = False
        has_negative = False
        
        review_segments = []
        labels = set()
        
        for aspect in chosen_aspects:
            # Randomly select if this aspect review is positive or negative
            is_pos = rng.choice([True, False])
            template_idx = rng.randint(0, 1) if is_pos else rng.randint(2, 3)
            review_segments.append(aspect_templates[aspect][template_idx])
            
            # Label naming convention
            labels.add(f"aspect:{aspect}")
            if is_pos:
                has_positive = True
            else:
                has_negative = True
                
        # Combine segments
        text = " ".join(review_segments)
        
        # Determine overall sentiment label
        if has_positive and has_negative:
            labels.add("sentiment:mixed")
        elif has_positive:
            labels.add("sentiment:positive")
        else:
            labels.add("sentiment:negative")

        rows.append({
            "id": f"rev-{i:05d}",
            "text": text,
            "labels": sorted(list(labels))
        })
        
    return rows


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_texts: list[str],
    batch_targets: torch.Tensor,
    vocab: dict[str, int],
) -> float:
    """Train the model for a single epoch."""
    model.train()
    indices, offsets = encode_batch(batch_texts, vocab)
    logits = model(indices, offsets)
    loss = loss_fn(logits, batch_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    texts: list[str],
    targets: torch.Tensor,
    vocab: dict[str, int],
    threshold: float = 0.35,
) -> dict[str, float]:
    """Evaluate model performance on the test set."""
    model.eval()
    indices, offsets = encode_batch(texts, vocab)
    logits = model(indices, offsets)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=torch.int64)
    return compute_metrics(targets.to(dtype=torch.int64), preds)


def run_demo_training(output_dir: str = "./output") -> None:
    """Orchestrate training of the local sentiment model, then quantize it."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Data
    reviews = make_synthetic_reviews(n=1000, seed=42)
    label_names = sorted(list({lbl for r in reviews for lbl in r["labels"]}))
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    
    # Train-test split
    random.seed(42)
    random.shuffle(reviews)
    split = int(len(reviews) * 0.8)
    train_rows, val_rows = reviews[:split], reviews[split:]
    
    train_texts = [r["text"] for r in train_rows]
    val_texts = [r["text"] for r in val_rows]
    
    def to_targets(batch_rows: list[dict[str, Any]]) -> torch.Tensor:
        y = torch.zeros((len(batch_rows), len(label_names)), dtype=torch.float32)
        for i, r in enumerate(batch_rows):
            for label in r["labels"]:
                y[i, label_to_idx[label]] = 1.0
        return y
        
    y_train = to_targets(train_rows)
    y_val = to_targets(val_rows)
    
    # 2. Build vocab and model
    vocab = build_vocab(train_texts, max_tokens=2000)
    model = SentimentAspectClassifier(vocab_size=len(vocab), num_labels=len(label_names))
    
    # 3. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    
    batch_size = 64
    epochs = 10
    print(f"Starting local training on {len(train_texts)} samples across {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for start in range(0, len(train_texts), batch_size):
            end = start + batch_size
            loss = train_epoch(
                model, 
                optimizer, 
                loss_fn, 
                train_texts[start:end], 
                y_train[start:end], 
                vocab
            )
            epoch_losses.append(loss)
        print(f"Epoch {epoch}/{epochs} | Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
        
    # 4. Evaluate Float Model
    float_metrics = evaluate(model, val_texts, y_val, vocab)
    print("\nFloat Model Evaluation:")
    print(json.dumps(float_metrics, indent=2))
    
    # 5. Quantize Model
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    
    quant_metrics = evaluate(quantized_model, val_texts, y_val, vocab)
    print("\nQuantized Model Evaluation:")
    print(json.dumps(quant_metrics, indent=2))
    
    # 6. Save Artifacts
    torch.save(model.state_dict(), out_path / "sentiment_model.pt")
    torch.save(quantized_model.state_dict(), out_path / "sentiment_model_quantized.pt")
    (out_path / "vocab.json").write_text(json.dumps(vocab, sort_keys=True))
    (out_path / "labels.json").write_text(json.dumps(label_names))
    
    float_size = (out_path / "sentiment_model.pt").stat().st_size
    quant_size = (out_path / "sentiment_model_quantized.pt").stat().st_size
    print(f"\nSaved models to {out_path}:")
    print(f"- Float Model: {float_size / 1024:.2f} KB")
    print(f"- Quantized Model: {quant_size / 1024:.2f} KB ({(1 - quant_size/float_size)*100:.1f}% reduction)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./pytorch_model/output")
    args = parser.parse_args()
    run_demo_training(args.output_dir)

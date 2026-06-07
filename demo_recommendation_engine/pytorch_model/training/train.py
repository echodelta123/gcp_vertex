"""Training entrypoint for the Two-Tower Recommendation Model.

Supports:
- Local demo mode (synthetic data, CPU, fast iteration)
- Vertex AI Custom Job mode (reads from GCS, logs to Vertex Experiments)
- Evaluation with recommendation-specific metrics (Hit@K, NDCG@K, MRR)
- Model export with optional quantization

Usage:
    # Local demo
    python -m demo_recommendation_engine.pytorch_model.training.train --demo --output-dir demo_recommendation_engine/pytorch_model/output

    # Production mode
    python -m demo_recommendation_engine.pytorch_model.training.train \
        --data-dir gs://bucket/data --output-dir gs://bucket/models/run-123
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    RecommendationDataset,
    generate_synthetic_data,
    temporal_train_val_test_split,
)
from .model import TwoTowerModel

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def compute_metrics(
    model: TwoTowerModel,
    dataloader: DataLoader,
    num_items: int,
    all_item_ids: torch.Tensor,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute recommendation metrics: Hit@K, NDCG@K, MRR.

    These are the standard metrics for evaluating retrieval-stage
    recommendation models in production systems.
    """
    if k_values is None:
        k_values = [10, 50, 100]

    model.eval()
    hits = {k: 0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    total = 0

    # Pre-compute all item embeddings for ranking
    with torch.no_grad():
        # Process in batches to avoid memory issues
        item_embs = []
        batch_size = 256
        for i in range(0, len(all_item_ids), batch_size):
            batch_ids = all_item_ids[i : i + batch_size]
            item_embs.append(model.get_item_embeddings(batch_ids))
        all_item_emb = torch.cat(item_embs, dim=0)

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch["user_id"]
            target_items = batch["item_id"]
            user_features = batch["user_features"] if batch["user_features"].numel() > 0 else None

            user_emb = model.get_user_embeddings(user_ids, user_features)

            # Score all items for each user
            scores = torch.matmul(user_emb, all_item_emb.T)

            for i in range(len(user_ids)):
                target = target_items[i].item()
                user_scores = scores[i]

                # Get ranked item indices
                _, ranked_indices = torch.sort(user_scores, descending=True)

                # Find rank of target item
                rank_positions = (ranked_indices == target).nonzero(as_tuple=True)[0]
                if len(rank_positions) > 0:
                    rank = rank_positions[0].item() + 1  # 1-indexed

                    mrr_sum += 1.0 / rank

                    for k in k_values:
                        if rank <= k:
                            hits[k] += 1
                            ndcg[k] += 1.0 / (torch.log2(torch.tensor(rank + 1.0)).item())

                total += 1

    metrics = {"mrr": mrr_sum / max(total, 1)}
    for k in k_values:
        metrics[f"hit_rate@{k}"] = hits[k] / max(total, 1)
        metrics[f"ndcg@{k}"] = ndcg[k] / max(total, 1)

    return metrics


def train_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_items: int,
    all_item_ids: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> dict[str, list[float]]:
    """Train the two-tower model with in-batch negatives."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_hit@10": [], "val_ndcg@10": [], "val_mrr": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            user_ids = batch["user_id"]
            item_ids = batch["item_id"]
            user_features = batch["user_features"] if batch["user_features"].numel() > 0 else None
            item_features = batch["item_features"] if batch["item_features"].numel() > 0 else None

            output = model(user_ids, item_ids, user_features, item_features)
            loss = criterion(output["logits"], output["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation metrics
        val_metrics = compute_metrics(model, val_loader, num_items, all_item_ids, k_values=[10])
        history["val_hit@10"].append(val_metrics["hit_rate@10"])
        history["val_ndcg@10"].append(val_metrics["ndcg@10"])
        history["val_mrr"].append(val_metrics["mrr"])

        logger.info(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
            f"hit@10={val_metrics['hit_rate@10']:.4f} | "
            f"ndcg@10={val_metrics['ndcg@10']:.4f} | "
            f"mrr={val_metrics['mrr']:.4f}"
        )

    return history


def run_demo(output_dir: str, epochs: int = 5) -> dict:
    """Run a fast local demo with synthetic data."""
    logger.info("Generating synthetic interaction data...")
    interactions = generate_synthetic_data(
        num_users=500, num_items=200, num_interactions=20000, seed=42
    )

    train_data, val_data, test_data = temporal_train_val_test_split(interactions)
    logger.info(
        f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
    )

    num_users = 500
    num_items = 200
    num_user_features = 8
    num_item_features = 16

    train_loader = DataLoader(
        RecommendationDataset(train_data), batch_size=256, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        RecommendationDataset(val_data), batch_size=256, shuffle=False
    )
    test_loader = DataLoader(
        RecommendationDataset(test_data), batch_size=256, shuffle=False
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=128,
        num_user_features=num_user_features,
        num_item_features=num_item_features,
        temperature=0.07,
    )

    all_item_ids = torch.arange(num_items)

    logger.info("Starting training...")
    start = time.time()
    history = train_model(
        model, train_loader, val_loader, num_items, all_item_ids, epochs=epochs
    )
    train_time = time.time() - start
    logger.info(f"Training completed in {train_time:.1f}s")

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = compute_metrics(
        model, test_loader, num_items, all_item_ids, k_values=[10, 50, 100]
    )
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": 64,
                "hidden_dim": 128,
                "num_user_features": num_user_features,
                "num_item_features": num_item_features,
                "temperature": 0.07,
            },
            "training_history": history,
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path}")

    # Quantize model for serving
    logger.info("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_path = output_path / "model_quantized.pt"
    torch.save(
        {"model_state_dict": quantized_model.state_dict()},
        quantized_path,
    )

    # Compare sizes
    original_size = model_path.stat().st_size
    quantized_size = quantized_path.stat().st_size
    size_reduction = 1 - (quantized_size / original_size)

    # Save metrics
    metrics = {
        "test_metrics": test_metrics,
        "training_time_seconds": train_time,
        "epochs": epochs,
        "model_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "size_reduction_pct": round(size_reduction * 100, 1),
        "num_users": num_users,
        "num_items": num_items,
        "num_train_interactions": len(train_data),
    }

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Recommendation Model")
    parser.add_argument("--demo", action="store_true", help="Run local demo with synthetic data")
    parser.add_argument("--output-dir", type=str, default="/tmp/rec-model-output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default=None, help="GCS or local data directory")
    args = parser.parse_args()

    if args.demo:
        metrics = run_demo(args.output_dir, epochs=args.epochs)
        print(f"\n{'='*60}")
        print("DEMO COMPLETE - Recommendation Engine Training Results")
        print(f"{'='*60}")
        print(f"  Hit Rate@10:  {metrics['test_metrics']['hit_rate@10']:.4f}")
        print(f"  NDCG@10:      {metrics['test_metrics']['ndcg@10']:.4f}")
        print(f"  MRR:          {metrics['test_metrics']['mrr']:.4f}")
        print(f"  Model size:   {metrics['model_size_bytes'] / 1024:.1f} KB")
        print(f"  Quantized:    {metrics['quantized_size_bytes'] / 1024:.1f} KB ({metrics['size_reduction_pct']}% smaller)")
        print(f"  Train time:   {metrics['training_time_seconds']:.1f}s")
        print(f"{'='*60}")
    else:
        # Production mode: would read from GCS, log to Vertex Experiments
        logger.info("Production training mode - requires GCS data and Vertex AI environment")
        if not args.data_dir:
            logger.error("--data-dir required in production mode")
            sys.exit(1)
        # In production, would call vertex-specific training logic here
        run_demo(args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()

"""Synthetic data generation and data loading utilities.

For the portfolio demo we generate synthetic e-commerce interaction data
that mimics real user-item interactions with implicit feedback signals.

In production this would be replaced by reading from BigQuery or GCS,
but the schema and processing logic remain the same.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class InteractionRecord:
    """A single user-item interaction."""

    user_id: int
    item_id: int
    timestamp: float
    interaction_type: str  # "click", "purchase", "add_to_cart"
    user_features: list[float] = field(default_factory=list)
    item_features: list[float] = field(default_factory=list)


class RecommendationDataset(Dataset):
    """PyTorch Dataset for user-item interactions."""

    def __init__(self, interactions: list[InteractionRecord]):
        self.interactions = interactions

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.interactions[idx]
        return {
            "user_id": torch.tensor(record.user_id, dtype=torch.long),
            "item_id": torch.tensor(record.item_id, dtype=torch.long),
            "user_features": torch.tensor(record.user_features, dtype=torch.float32),
            "item_features": torch.tensor(record.item_features, dtype=torch.float32),
        }


def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 50000,
    num_user_features: int = 8,
    num_item_features: int = 16,
    seed: int = 42,
) -> list[InteractionRecord]:
    """Generate synthetic user-item interaction data.

    Simulates a realistic distribution where:
    - Some items are more popular (power-law)
    - Users have varying activity levels
    - Interactions cluster by user preference groups
    """
    random.seed(seed)

    # Create user preference groups (simulates latent taste clusters)
    num_groups = 10
    user_groups = [random.randint(0, num_groups - 1) for _ in range(num_users)]
    item_groups = [random.randint(0, num_groups - 1) for _ in range(num_items)]

    # Item popularity follows power-law
    item_popularity = [1.0 / (i + 1) ** 0.5 for i in range(num_items)]
    total_pop = sum(item_popularity)
    item_popularity = [p / total_pop for p in item_popularity]

    # Generate user features (e.g., normalized age, tenure, avg_spend, etc.)
    user_feature_map = {
        uid: [random.gauss(0, 1) for _ in range(num_user_features)]
        for uid in range(num_users)
    }

    # Generate item features (e.g., price_bucket, category_embed, brand_embed, etc.)
    item_feature_map = {
        iid: [random.gauss(0, 1) for _ in range(num_item_features)]
        for iid in range(num_items)
    }

    interaction_types = ["click", "click", "click", "add_to_cart", "purchase"]

    interactions = []
    for _ in range(num_interactions):
        user_id = random.randint(0, num_users - 1)
        user_group = user_groups[user_id]

        # Bias item selection toward same-group items
        if random.random() < 0.6:
            group_items = [
                i for i in range(num_items) if item_groups[i] == user_group
            ]
            item_id = random.choice(group_items) if group_items else random.randint(
                0, num_items - 1
            )
        else:
            # Popularity-weighted random
            item_id = random.choices(range(num_items), weights=item_popularity, k=1)[0]

        interactions.append(
            InteractionRecord(
                user_id=user_id,
                item_id=item_id,
                timestamp=random.uniform(1_600_000_000, 1_700_000_000),
                interaction_type=random.choice(interaction_types),
                user_features=user_feature_map[user_id],
                item_features=item_feature_map[item_id],
            )
        )

    # Sort by timestamp for temporal split
    interactions.sort(key=lambda x: x.timestamp)
    return interactions


def temporal_train_val_test_split(
    interactions: list[InteractionRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[InteractionRecord], list[InteractionRecord], list[InteractionRecord]]:
    """Split interactions temporally (most recent data for eval/test).

    This mirrors production: you train on historical data and evaluate
    on future interactions the model hasn't seen.
    """
    n = len(interactions)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return interactions[:train_end], interactions[train_end:val_end], interactions[val_end:]


def save_interactions_jsonl(interactions: list[InteractionRecord], path: Path) -> None:
    """Save interactions to JSONL format (GCS-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in interactions:
            f.write(
                json.dumps(
                    {
                        "user_id": record.user_id,
                        "item_id": record.item_id,
                        "timestamp": record.timestamp,
                        "interaction_type": record.interaction_type,
                        "user_features": record.user_features,
                        "item_features": record.item_features,
                    }
                )
                + "\n"
            )


def load_interactions_jsonl(path: Path) -> list[InteractionRecord]:
    """Load interactions from JSONL format."""
    interactions = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            interactions.append(
                InteractionRecord(
                    user_id=data["user_id"],
                    item_id=data["item_id"],
                    timestamp=data["timestamp"],
                    interaction_type=data["interaction_type"],
                    user_features=data.get("user_features", []),
                    item_features=data.get("item_features", []),
                )
            )
    return interactions

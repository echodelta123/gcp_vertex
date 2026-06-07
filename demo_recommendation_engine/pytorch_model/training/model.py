"""Two-Tower Recommendation Model.

Architecture:
- User Tower: Encodes user features (history, demographics) into a dense embedding.
- Item Tower: Encodes item features (category, attributes) into a dense embedding.
- Scoring: Dot-product similarity between user and item embeddings.

This architecture is production-proven at scale (YouTube, Google Play, Pinterest)
and supports efficient approximate nearest-neighbor retrieval at serving time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user features into a fixed-size embedding."""

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_user_features: int = 0,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        input_dim = embedding_dim + num_user_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self, user_ids: torch.Tensor, user_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        if user_features is not None and user_features.shape[-1] > 0:
            x = torch.cat([x, user_features], dim=-1)
        else:
            # Pad with zeros to match expected input dimension
            pad = torch.zeros(
                x.shape[0], self.mlp[0].in_features - x.shape[-1], device=x.device
            )
            x = torch.cat([x, pad], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item features into a fixed-size embedding."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_item_features: int = 0,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        input_dim = embedding_dim + num_item_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self, item_ids: torch.Tensor, item_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        if item_features is not None and item_features.shape[-1] > 0:
            x = torch.cat([x, item_features], dim=-1)
        else:
            # Pad with zeros to match expected input dimension
            pad = torch.zeros(
                x.shape[0], self.mlp[0].in_features - x.shape[-1], device=x.device
            )
            x = torch.cat([x, pad], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-tower recommendation model with contrastive learning.

    Training uses in-batch negatives (sampled softmax) which is both efficient
    and effective for large-scale recommendation systems.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_user_features: int = 0,
        num_item_features: int = 0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users, embedding_dim, hidden_dim, num_user_features
        )
        self.item_tower = ItemTower(
            num_items, embedding_dim, hidden_dim, num_item_features
        )
        self.temperature = temperature
        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
        item_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)

        # Compute similarity matrix (in-batch negatives)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        return {
            "logits": logits,
            "labels": labels,
            "user_embeddings": user_emb,
            "item_embeddings": item_emb,
        }

    def get_user_embeddings(
        self, user_ids: torch.Tensor, user_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_features)

    def get_item_embeddings(
        self, item_ids: torch.Tensor, item_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.item_tower(item_ids, item_features)

# src/models/MFBPR.py
import torch
import torch.nn as nn


class MFBPR(nn.Module):
    """
    Matrix Factorization trained with BPR loss.
    Score(u,i) = <p_u, q_i>
    """
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)         # [B, d]
        v = self.item_emb(items)         # [B, d]
        return (u * v).sum(dim=1)        # [B]

    def forward(self, users, pos_items, neg_items):
        pos_scores = self.score(users, pos_items)
        neg_scores = self.score(users, neg_items)
        return pos_scores, neg_scores

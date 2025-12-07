# src/models/lightgcn.py

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN cho implicit feedback.

    Nodes:
      0 .. num_users-1                      : user nodes
      num_users .. num_users+num_items-1    : item nodes
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = 64,
        n_layers: int = 3,
        reg_weight: float = 1e-4,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items

        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        # Embedding gốc (ego embedding) cho cả user + item
        self.embedding = nn.Embedding(self.num_nodes, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    # ---------------------- Core propagation ---------------------- #
    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        """
        LightGCN propagation:
            x^(0) = Embedding
            x^(k+1) = A_hat @ x^(k)
            final = mean_k x^(k)

        Args:
            adj: normalized adjacency (sparse) [num_nodes, num_nodes]

        Returns:
            all_embeddings: [num_nodes, emb_dim]
        """
        x = self.embedding.weight  # [N, d]
        embs = [x]

        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)

        embs_stack = torch.stack(embs, dim=0)   # [K+1, N, d]
        all_embeddings = embs_stack.mean(dim=0) # [N, d]
        return all_embeddings

    def get_user_item_embeddings(
        self, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience function:
        Returns:
            user_embs: [num_users, emb_dim]
            item_embs: [num_items, emb_dim]
        """
        all_embs = self.propagate(adj)
        user_embs = all_embs[:self.num_users]
        item_embs = all_embs[self.num_users:]
        return user_embs, item_embs

    # ---------------------- BPR Loss (version tối ưu) ---------------------- #
    def bpr_loss_from_all_embs(
        self,
        all_embs: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        """
        Version dùng khi đã có all_embs = propagate(adj).

        Args:
            all_embs: [num_nodes, emb_dim]
            users: [B] user indices (0 .. num_users-1)
            pos_items: [B] positive item indices (0 .. num_items-1)
            neg_items: [B] negative item indices (0 .. num_items-1)
        """
        # Embedding sau khi propagate (dùng để tính score)
        user_embs = all_embs[users]  # [B, d]
        pos_embs = all_embs[pos_items + self.num_users]
        neg_embs = all_embs[neg_items + self.num_users]

        # Inner product scores
        pos_scores = torch.sum(user_embs * pos_embs, dim=-1)  # [B]
        neg_scores = torch.sum(user_embs * neg_embs, dim=-1)  # [B]

        # BPR Loss
        diff = pos_scores - neg_scores
        bpr = -F.logsigmoid(diff).mean()

        # Regularization trên ego embedding (gốc) – đúng paper
        ego_users = self.embedding.weight[users]
        ego_pos = self.embedding.weight[pos_items + self.num_users]
        ego_neg = self.embedding.weight[neg_items + self.num_users]

        reg = (
            ego_users.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / users.shape[0]

        loss = bpr + self.reg_weight * reg

        log_dict = {
            "bpr": bpr.item(),
            "reg": reg.item(),
            "loss": loss.item(),
        }
        return loss, log_dict

    def bpr_loss_slow(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        adj: torch.Tensor,
    ):
        """
        Version an toàn cho mini-batch:
          - Mỗi batch: propagate(adj) một lần
          - Không bị stale embeddings

        Args:
            users, pos_items, neg_items: [B]
            adj: normalized adjacency (sparse)

        Returns:
            loss, log_dict
        """
        all_embs = self.propagate(adj)
        return self.bpr_loss_from_all_embs(
            all_embs=all_embs,
            users=users,
            pos_items=pos_items,
            neg_items=neg_items,
        )

    # ---------------------- Full ranking (eval) ---------------------- #
    def full_sort_scores(
        self,
        users: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính điểm (score) cho một batch user trên toàn bộ item.
        Args:
            users: [B] user indices
            adj: normalized adjacency

        Returns:
            scores: [B, num_items]
        """
        user_embs, item_embs = self.get_user_item_embeddings(adj)
        u_emb = user_embs[users]          # [B, d]
        scores = torch.matmul(u_emb, item_embs.t())  # [B, num_items]
        return scores

# src/models/NGCF.py

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    """
    PyTorch implementation of NGCF for implicit CF, thiết kế để:
      - dùng chung dataloader + train loop BPR với LightGCN
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = 64,
        layer_sizes=None,
        reg_weight: float = 1e-4,
        mess_dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [64, 64]

        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.reg_weight = reg_weight
        self.mess_dropout = mess_dropout
        self.leaky_relu_slope = leaky_relu_slope

        # ===== 1. Ego embedding (Layer 0) =====
        # Một bảng embedding chung cho user + item
        self.embedding = nn.Embedding(num_users + num_items, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

        # ===== 2. NGCF weights cho từng layer =====
        # kích thước: [d_k, d_{k+1}], với d_0 = emb_dim
        dims = [emb_dim] + list(layer_sizes)

        self.W_gc = nn.ParameterList()
        self.b_gc = nn.ParameterList()
        self.W_bi = nn.ParameterList()
        self.b_bi = nn.ParameterList()

        for k in range(self.n_layers):
            d_in, d_out = dims[k], dims[k + 1]

            W_gc_k = nn.Parameter(torch.empty(d_in, d_out))
            b_gc_k = nn.Parameter(torch.zeros(1, d_out))
            W_bi_k = nn.Parameter(torch.empty(d_in, d_out))
            b_bi_k = nn.Parameter(torch.zeros(1, d_out))

            nn.init.xavier_uniform_(W_gc_k)
            nn.init.xavier_uniform_(W_bi_k)

            self.W_gc.append(W_gc_k)
            self.b_gc.append(b_gc_k)
            self.W_bi.append(W_bi_k)
            self.b_bi.append(b_bi_k)

    # ------------------------------------------------------------------
    #  Helper: tách user/item embedding từ all_embs
    # ------------------------------------------------------------------
    def split_user_item(self, all_embs: torch.Tensor):
        """
        all_embs: (num_users+num_items, D_total)
        return: user_embs, item_embs
        """
        user_embs = all_embs[: self.num_users, :]
        item_embs = all_embs[self.num_users :, :]
        return user_embs, item_embs

    # ------------------------------------------------------------------
    #  Propagation NGCF
    # ------------------------------------------------------------------
    def propagate_no_grad(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Convenience dùng cho eval: propagate không grad.
        """
        self.eval()
        with torch.no_grad():
            return self._propagate_impl(adj)

    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Dùng trong training/eval có grad.
        adj: torch.sparse_coo_tensor, shape (N, N) với N = num_users + num_items
        return: all_embs (N, D_total) – concat embedding qua các layer.
        """
        return self._propagate_impl(adj)

    def _propagate_impl(self, adj: torch.Tensor) -> torch.Tensor:
        # ego_embeddings: (N, d0)
        ego_embeddings = self.embedding.weight
        all_embeddings = [ego_embeddings]

        x = ego_embeddings
        for k in range(self.n_layers):
            # 1) Sum messages từ neighbour: A * x
            side_embeddings = torch.sparse.mm(adj, x)  # (N, d_k)

            # 2) Sum part: LeakyReLU( side @ W_gc + b_gc )
            sum_embeddings = torch.matmul(side_embeddings, self.W_gc[k]) + self.b_gc[k]
            sum_embeddings = F.leaky_relu(
                sum_embeddings, negative_slope=self.leaky_relu_slope
            )

            # 3) Bi-interaction: LeakyReLU( (x ⊙ side) @ W_bi + b_bi )
            bi = x * side_embeddings
            bi_embeddings = torch.matmul(bi, self.W_bi[k]) + self.b_bi[k]
            bi_embeddings = F.leaky_relu(
                bi_embeddings, negative_slope=self.leaky_relu_slope
            )

            # 4) Tổng message
            x = sum_embeddings + bi_embeddings

            # 5) Message dropout
            if self.training and self.mess_dropout > 0.0:
                x = F.dropout(x, p=self.mess_dropout, training=True)

            # 6) Lưu embedding layer k (không L2-normalize)
            all_embeddings.append(x)

        # concat theo chiều feature, giống code TF
        all_embs = torch.cat(all_embeddings, dim=1)
        return all_embs

    # ------------------------------------------------------------------
    #  BPR loss (slow – gọi propagate mỗi batch)
    # ------------------------------------------------------------------
    def bpr_loss_slow(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        users, pos_items, neg_items: LongTensor trên cùng device
        adj: sparse adj trên cùng device
        """
        all_embs = self.propagate(adj)  # (N, D_total)
        return self.bpr_loss_from_all_embs(all_embs, users, pos_items, neg_items)

    def bpr_loss_from_all_embs(
        self,
        all_embs: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Tách embedding user/item từ all_embs rồi tính BPR + L2.
        """
        user_all, item_all = self.split_user_item(all_embs)

        # ===== 1. Embedding sau propagate (dùng cho BPR) =====
        u_g = user_all[users]          # (B, D_total)
        pos_g = item_all[pos_items]    # (B, D_total)
        neg_g = item_all[neg_items]    # (B, D_total)

        # BPR loss
        pos_scores = (u_g * pos_g).sum(dim=1)  # (B,)
        neg_scores = (u_g * neg_g).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # ===== 2. Regularization trên ego embedding (Layer 0) =====
        ego_all = self.embedding.weight
        ego_user = ego_all[users]
        ego_pos = ego_all[pos_items + self.num_users]
        ego_neg = ego_all[neg_items + self.num_users]

        reg = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / users.shape[0]

        reg = 0.5 * reg  # optional có thể bỏ 0.5

        loss = bpr + self.reg_weight * reg

        log = {
            "bpr": float(bpr.detach().cpu().item()),
            "reg": float(reg.detach().cpu().item()),
        }
        return loss, log

    # ------------------------------------------------------------------
    #  Scoring cho eval: full ranking trên tất cả item
    # ------------------------------------------------------------------
    def full_sort_scores(
        self,
        all_embs: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        """
        all_embs: (N, D_total) từ propagate(adj)
        users: (B,)
        return: scores (B, num_items)
        """
        user_all, item_all = self.split_user_item(all_embs)
        u = user_all[users]  # (B, D_total)
        scores = torch.matmul(u, item_all.t())  # (B, num_items)
        return scores

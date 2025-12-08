# src/data_utils/graph_builder_time_decay.py

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


class TimeDecayGraphBuilder:
    """
    Build normalized adjacency matrix with time-decay edge weights.

    Nodes:
      0 .. num_users-1                      : user nodes
      num_users .. num_users+num_items-1    : item nodes

    Edges:
      undirected, weighted:
        - từ file CSV (u, v, weight) đã time-decay + (optional) user-normalized
        - nếu add_self_loop=True: thêm self-loop A + I (trọng số = 1.0 cho mỗi node)

    Chuẩn hoá:
      A_hat = D^{-1/2} A D^{-1/2}
      với A_ij = weight_ij (time-decay).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        weight_csv: str | Path,
        add_self_loop: bool = False,
        verbose: bool = True,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items

        self.weight_csv = Path(weight_csv)
        self.add_self_loop = add_self_loop
        self.verbose = verbose

        self._adj_norm: Optional[torch.Tensor] = None  # cache

    def build_normalized_adj(self, device: str = "cpu") -> torch.Tensor:
        """
        Build symmetric normalized adjacency:
            A_hat = D^{-1/2} A D^{-1/2}

        Returns:
            torch.sparse.FloatTensor (num_nodes, num_nodes)
        """
        if self._adj_norm is not None:
            return self._adj_norm.to(device)

        # ========= 1) Load time-decay edges =========
        edges = pd.read_csv(self.weight_csv)

        if self.verbose:
            print("=== TimeDecayGraphBuilder ===")
            print(f"CSV path     : {self.weight_csv}")
            print(f"#Edges (u,v) : {len(edges):,}")
            print(f"u range      : {edges['u'].min()} .. {edges['u'].max()}")
            print(f"v range      : {edges['v'].min()} .. {edges['v'].max()}")
            print(f"weight stats : min={edges['weight'].min():.6f}, "
                  f"max={edges['weight'].max():.6f}, "
                  f"mean={edges['weight'].mean():.6f}")
            print(f"add_self_loop: {self.add_self_loop}")
            print("=============================\n")

        u = edges["u"].to_numpy(dtype=np.int64)
        v = edges["v"].to_numpy(dtype=np.int64)
        w = edges["weight"].to_numpy(dtype=np.float32)

        offset = self.num_users
        n_edges = len(edges)

        # Số edge trong adjacency:
        #  - user->item
        #  - item->user
        #  - (optional) self-loop
        num_edges = n_edges * 2 + (self.num_nodes if self.add_self_loop else 0)

        row = np.empty(num_edges, dtype=np.int64)
        col = np.empty(num_edges, dtype=np.int64)
        data = np.empty(num_edges, dtype=np.float32)

        idx = 0

        # --- user -> item ---
        row[idx: idx + n_edges] = u
        col[idx: idx + n_edges] = v + offset
        data[idx: idx + n_edges] = w
        idx += n_edges

        # --- item -> user ---
        row[idx: idx + n_edges] = v + offset
        col[idx: idx + n_edges] = u
        data[idx: idx + n_edges] = w
        idx += n_edges

        # --- optional self-loop A + I ---
        if self.add_self_loop:
            nodes = np.arange(self.num_nodes, dtype=np.int64)
            row[idx: idx + self.num_nodes] = nodes
            col[idx: idx + self.num_nodes] = nodes
            data[idx: idx + self.num_nodes] = 1.0
            idx += self.num_nodes

        if idx < num_edges:
            # phòng trường hợp hiếm khi có thiếu
            row = row[:idx]
            col = col[:idx]
            data = data[:idx]

        # ========= 2) Degree with weights =========
        # deg_i = sum_j A_ij (bao gồm weight time-decay + self-loop nếu có)
        deg = np.bincount(
            row, weights=data, minlength=self.num_nodes
        ).astype(np.float32)

        d_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        mask = deg > 0
        d_inv_sqrt[mask] = np.power(deg[mask], -0.5)

        # ========= 3) Normalization =========
        # data_norm_ij = d^{-1/2}_i * A_ij * d^{-1/2}_j
        data_norm = d_inv_sqrt[row] * data * d_inv_sqrt[col]

        indices = np.vstack([row, col])  # (2, nnz)
        indices_t = torch.from_numpy(indices).long()
        values_t = torch.from_numpy(data_norm).float()
        shape = (self.num_nodes, self.num_nodes)

        adj_norm = torch.sparse_coo_tensor(indices_t, values_t, shape)
        adj_norm = adj_norm.coalesce()

        self._adj_norm = adj_norm  # cache on CPU
        return adj_norm.to(device)

# src/data_utils/graph_builder.py

from typing import Dict, List
import numpy as np
import torch


class GraphBuilder:
    """
    Build normalized adjacency matrix for LightGCN/NGCF-style models.

    Nodes:
      0 .. num_users-1                      : user nodes
      num_users .. num_users+num_items-1    : item nodes

    Edges:
      undirected, binary (no weights): user <-> item if interaction exists in train.
      (tuỳ chọn) self-loop: i <-> i nếu add_self_loop=True (dùng cho NGCF).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_user_items: Dict[int, List[int]],
        add_self_loop: bool = False,
    ):
        """
        Args:
            num_users: total number of users (indexed [0, num_users-1])
            num_items: total number of items (indexed [0, num_items-1])
            train_user_items: dict {user: [item1, item2, ...]} from train.txt
            add_self_loop: nếu True -> dùng A + I (cho NGCF); False -> chỉ A (cho LightGCN).
        """
        self.num_users = num_users
        self.num_items = num_items
        self.train_user_items = train_user_items
        self.add_self_loop = add_self_loop

        self.num_nodes = self.num_users + self.num_items
        self._adj_norm = None  # cache after first build

    def build_normalized_adj(self, device: str = "cpu") -> torch.Tensor:
        """
        Build symmetric normalized adjacency matrix:
            A_hat = D^{-1/2} A D^{-1/2}
        where A is the bipartite adjacency (user-item graph),
        optionally with self-loop (A + I) nếu add_self_loop=True.

        Returns:
            torch.sparse.FloatTensor of shape (num_nodes, num_nodes)
        """
        if self._adj_norm is not None:
            return self._adj_norm.to(device)

        # -------- 1) Collect all edges (undirected) --------
        # For each (u, i) in train, add:
        #   u -> i_node, i_node -> u
        num_interactions = sum(len(v) for v in self.train_user_items.values())
        num_edges = num_interactions * 2  # undirected (bidirectional)

        row = np.empty(num_edges, dtype=np.int64)
        col = np.empty(num_edges, dtype=np.int64)
        data = np.ones(num_edges, dtype=np.float32)

        offset = self.num_users  # item node index offset
        idx = 0
        for u, items in self.train_user_items.items():
            if not items:
                continue
            items_arr = np.asarray(items, dtype=np.int64)
            m = len(items_arr)

            # u -> item
            row[idx: idx + m] = u
            col[idx: idx + m] = items_arr + offset

            # item -> u
            row[idx + m: idx + 2 * m] = items_arr + offset
            col[idx + m: idx + 2 * m] = u

            idx += 2 * m

        # Nếu vì lý do nào đó idx < num_edges (user rỗng), cắt lại:
        if idx < num_edges:
            row = row[:idx]
            col = col[:idx]
            data = data[:idx]

        # -------- 1b) Optional: thêm self-loop cho mọi node (A + I) --------
        if self.add_self_loop:
            self_loop_nodes = np.arange(self.num_nodes, dtype=np.int64)
            row = np.concatenate([row, self_loop_nodes])
            col = np.concatenate([col, self_loop_nodes])
            data = np.concatenate(
                [data, np.ones(self.num_nodes, dtype=np.float32)]
            )

        # -------- 2) Compute node degrees --------
        # deg_i = số edges đi ra từ node i
        deg = np.bincount(row, minlength=self.num_nodes).astype(np.float32)

        # D^{-1/2}, tránh chia cho 0
        d_inv_sqrt = np.zeros_like(deg)
        mask = deg > 0
        d_inv_sqrt[mask] = np.power(deg[mask], -0.5)

        # -------- 3) Apply normalization --------
        data_norm = d_inv_sqrt[row] * d_inv_sqrt[col]

        # -------- 4) Convert to torch.sparse tensor --------
        indices = np.vstack([row, col])  # shape (2, nnz)

        indices_t = torch.from_numpy(indices).long()
        values_t = torch.from_numpy(data_norm).float()
        shape = (self.num_nodes, self.num_nodes)

        adj_norm = torch.sparse_coo_tensor(indices_t, values_t, shape)
        adj_norm = adj_norm.coalesce()  # ensure canonical form

        self._adj_norm = adj_norm  # cache on CPU by default

        return adj_norm.to(device)

# src/data_utils/dataloader.py

from pathlib import Path

class TxtCFDataLoader:
    """
    Generic loader cho các dataset CF implicit
    dùng format LightGCN-style:
      - train.txt: user item1 item2 ...
      - val.txt  : user item_val
      - test.txt : user item_test
    """

    def __init__(self, data_dir, verbose: bool = True):
        self.data_dir = Path(data_dir)

        # 1) Load từng file
        self.train = self._load_interactions(self.data_dir / "train.txt")
        self.val   = self._load_interactions(self.data_dir / "val.txt")
        self.test  = self._load_interactions(self.data_dir / "test.txt")

        # 2) Suy ra num_users, num_items từ cả 3 split
        self.num_users = self._infer_num_users()
        self.num_items = self._infer_num_items()

        # 3) Đếm số interactions
        self.num_interactions_train = sum(len(items) for items in self.train.values())
        self.num_interactions_val   = sum(len(items) for items in self.val.values())
        self.num_interactions_test  = sum(len(items) for items in self.test.values())

        if verbose:
            self._print_stats()

    def _load_interactions(self, path: Path):
        user_dict = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                u = int(parts[0])
                items = list(map(int, parts[1:]))
                user_dict[u] = items
        return user_dict

    def _infer_num_users(self) -> int:
        all_users = set(self.train.keys()) | set(self.val.keys()) | set(self.test.keys())
        return max(all_users) + 1 if all_users else 0

    def _infer_num_items(self) -> int:
        mx = -1
        for d in [self.train, self.val, self.test]:
            for items in d.values():
                if items:
                    mx = max(mx, max(items))
        return mx + 1 if mx >= 0 else 0

    def _print_stats(self):
        print("=== TxtCFDataLoader ===")
        print(f"Data dir          : {self.data_dir}")
        print(f"#Users            : {self.num_users}")
        print(f"#Items            : {self.num_items}")
        print(f"#Train interactions: {self.num_interactions_train}")
        print(f"#Val interactions  : {self.num_interactions_val}")
        print(f"#Test interactions : {self.num_interactions_test}")
        print("========================")

    # API tiện lợi cho model
    def get_train_pos(self):
        return self.train

    def get_val_truth(self):
        # mỗi user có 1 item val
        return {u: items[0] for u, items in self.val.items() if items}

    def get_test_truth(self):
        return {u: items[0] for u, items in self.test.items() if items}

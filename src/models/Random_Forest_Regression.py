# Cao Minh Đạt - 23162015

import numpy as np
import pandas as pd

def train_test_split_np(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = int(len(X) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - u / v) if v != 0 else 0.0



class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features  
        self.rng = np.random.default_rng(random_state)
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X], dtype=float)

    def _predict_one(self, x, node):
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _stop(self, y, depth, n_samples):
        if n_samples < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if np.allclose(y, y[0]):
            return True
        return False

    def _feature_subset(self, n_features):
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif isinstance(mf, int):
            k = max(1, min(n_features, mf))
        else:
            k = n_features
        return self.rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y, feature_indices):
        n_samples, _ = X.shape

        best_feat, best_thr = None, None
        best_loss = np.inf

        y_sum = y.sum()
        y_sq_sum = (y * y).sum()

        def sse(count, sum_, sq_sum):
            if count <= 0:
                return 0.0
            return float(sq_sum - (sum_ * sum_) / count)

        for f in feature_indices:
            x = X[:, f]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            unique_mask = np.diff(x_sorted) != 0
            if not np.any(unique_mask):
                continue

            left_count = 0
            left_sum = 0.0
            left_sq_sum = 0.0

            for i in range(0, n_samples - 1):
                yi = y_sorted[i]
                left_count += 1
                left_sum += yi
                left_sq_sum += yi * yi

                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                right_count = n_samples - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                right_sum = y_sum - left_sum
                right_sq_sum = y_sq_sum - left_sq_sum

                loss = sse(left_count, left_sum, left_sq_sum) + sse(right_count, right_sum, right_sq_sum)
                if loss < best_loss:
                    best_loss = loss
                    best_feat = f
                    best_thr = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        return best_feat, best_thr

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if self._stop(y, depth, n_samples):
            return _TreeNode(value=float(np.mean(y)))

        feature_indices = self._feature_subset(n_features)
        feat, thr = self._best_split(X, y, feature_indices)

        if feat is None:
            return _TreeNode(value=float(np.mean(y)))

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return _TreeNode(value=float(np.mean(y)))

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(feature=feat, threshold=float(thr), left=left, right=right)

class RandomForestRegressorScratch:
    def __init__(
        self,
        n_estimators=50,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.rng = np.random.default_rng(random_state)
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples = len(X)
        self.trees = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.integers(0, n_samples, size=n_samples)
                Xb = X[idx]
                yb = y[idx]
            else:
                Xb, yb = X, y

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(self.rng.integers(0, 1_000_000_000)),
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.stack([t.predict(X) for t in self.trees], axis=0)  # (n_trees, n_samples)
        return np.mean(preds, axis=0)

if __name__ == "__main__":
    data = pd.read_csv("ratings_small.csv")

    X = data[["userId", "movieId"]].values
    y = data["rating"].values

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, seed=42)

    model = RandomForestRegressorScratch(
        n_estimators=100,        
        max_depth=20,           
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MSE:", mse(y_test, y_pred))
    print("R2 :", r2_score_np(y_test, y_pred))

    sample = np.array([[1, 31]], dtype=float)
    print("Predicted rating :", float(model.predict(sample)[0]))


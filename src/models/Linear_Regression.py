# Phạm Lê Anh Duy - 20162012
import os
import numpy as np
import pandas as pd
class Linear_Regression:


    def __init__(self, learning_rate=0.01, n_iterations=1000, use_standardize=True):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.use_standardize = bool(use_standardize)

        self.weights = None
        self.bias = None

        self.x_mean_ = None
        self.x_std_ = None

    def _fit_scaler(self, X):
        self.x_mean_ = np.mean(X, axis=0)
        self.x_std_ = np.std(X, axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0  

    def _transform(self, X):
        if not self.use_standardize:
            return X
        if self.x_mean_ is None or self.x_std_ is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        if self.use_standardize:
            self._fit_scaler(X)
            X_train = self._transform(X)
        else:
            X_train = X

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = X_train @ self.weights + self.bias
            error = y_pred - y

            dw = (2.0 / n_samples) * (X_train.T @ error)
            db = (2.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        if self.weights is None or self.bias is None:
            if self.use_standardize:
                self._fit_scaler(X)
            self.weights = np.zeros(n_features, dtype=float)
            self.bias = 0.0

        Xb = self._transform(X) if self.use_standardize else X

        y_pred = Xb @ self.weights + self.bias
        error = y_pred - y

        dw = (2.0 / n_samples) * (Xb.T @ error)
        db = (2.0 / n_samples) * np.sum(error)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return self

    def batch_fit(self, X, y, batch_size=32):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        if self.use_standardize:
            self._fit_scaler(X)
            X_all = self._transform(X)
        else:
            X_all = X

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                Xb = X_all[start:end]
                yb = y[start:end]

                m = len(yb)
                if m == 0:
                    continue

                y_pred = Xb @ self.weights + self.bias
                error = y_pred - yb

                dw = (2.0 / m) * (Xb.T @ error)
                db = (2.0 / m) * np.sum(error)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xp = self._transform(X) if self.use_standardize else X
        return Xp @ self.weights + self.bias

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def score(self, X, y):
        """R^2 score"""
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return float(1.0 - u / v) if v != 0 else 0.0

    def get_params(self):
        return {
            "weights": self.weights,
            "bias": self.bias,
            "x_mean_": self.x_mean_,
            "x_std_": self.x_std_,
            "use_standardize": self.use_standardize
        }

    def set_params(self, params):
        self.weights = params.get("weights", self.weights)
        self.bias = params.get("bias", self.bias)
        self.x_mean_ = params.get("x_mean_", self.x_mean_)
        self.x_std_ = params.get("x_std_", self.x_std_)
        self.use_standardize = params.get("use_standardize", self.use_standardize)
        return self

    def save_model(self, file_path):
        params = self.get_params()
        np.savez(
            file_path,
            weights=params["weights"],
            bias=params["bias"],
            x_mean_=params["x_mean_"],
            x_std_=params["x_std_"],
            use_standardize=np.array([int(params["use_standardize"])], dtype=int)
        )

    def load_model(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        params = {
            "weights": data["weights"],
            "bias": float(data["bias"]),
            "x_mean_": data["x_mean_"] if "x_mean_" in data else None,
            "x_std_": data["x_std_"] if "x_std_" in data else None,
            "use_standardize": bool(int(data["use_standardize"][0])) if "use_standardize" in data else True
        }
        return self.set_params(params)

    def ridge_fit(self, X, y, alpha=1.0):
        """
        Ridge regression via GD:
        Loss = MSE + alpha * ||w||^2
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        if self.use_standardize:
            self._fit_scaler(X)
            X_train = self._transform(X)
        else:
            X_train = X

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        alpha = float(alpha)

        for _ in range(self.n_iterations):
            y_pred = X_train @ self.weights + self.bias
            error = y_pred - y

            dw = (2.0 / n_samples) * (X_train.T @ error) + 2.0 * alpha * self.weights
            db = (2.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def reset(self):
        self.weights = None
        self.bias = None
        self.x_mean_ = None
        self.x_std_ = None
        return self

    def __str__(self):
        return f"Linear_Regression(lr={self.learning_rate}, iters={self.n_iterations}, standardize={self.use_standardize})"

    def __repr__(self):
        return self.__str__()
if __name__ == "__main__":


 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "ratings_small.csv")

    data = pd.read_csv(DATA_PATH)

    X = data[["userId", "movieId"]].values
    y = data["rating"].values

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    model = Linear_Regression(
        learning_rate=0.01,
        n_iterations=1000
    )

    model.fit(X, y)

    print("MSE:", model.evaluate(X, y))
    print("R2 :", model.score(X, y))

    sample = np.array([[1, 31]])
    sample = (sample - X_mean) / X_std

    print("Predicted rating:", model.predict(sample)[0])
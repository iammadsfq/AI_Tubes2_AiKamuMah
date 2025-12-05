import numpy as np
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin


class SVMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, mode='one-vs-all'):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.mode = mode  # 'one-vs-all'
        self.models = []  # Menyimpan sub-model untuk multiclass
        self.mean = 0
        self.std = 0

    def compute_scaler(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1

    def transform(self, X):
        # Standardisasi
        return (X - self.mean) / self.std

    def fit(self, X, y):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        if len(y.shape) > 1:
            y = y.ravel()
        
        self.compute_scaler(X)
        X = self.transform(X)

        classes = np.unique(y)

        if self.mode == 'one-vs-all':
            for c in classes:
                y_binary = np.where(y == c, 1, -1)
                model = SVM_Binary(self.learning_rate, self.lambda_param, self.n_iters)
                model.fit(X, y_binary)
                self.models.append((c, model))

    def predict(self, X):
        X = self.transform(X)

        if self.mode == 'one-vs-all':
            scores = []
            for (c, model) in self.models:
                scores.append(model.decision_function(X))
            scores = np.array(scores)
            idx = np.argmax(scores, axis=0)
            return np.array([self.models[i][0] for i in idx])

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class SVM_Binary:
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            margin = y * (np.dot(X, self.w) + self.b)
            misclassified_idx = np.where(margin < 1)[0]

            if len(misclassified_idx) > 0:
                miss_X = X[misclassified_idx]
                miss_y = y[misclassified_idx]

                loss_dw = -np.dot(miss_X.T, miss_y)
                loss_db = -np.sum(miss_y)
            else:
                loss_dw = 0
                loss_db = 0

            grad_w = 2 * self.lambda_param * self.w + (1 / n_samples) * loss_dw
            grad_b = (1 / n_samples) * loss_db

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def main():
    # Ini buat ngetes aja
    X = np.array([
        [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
        [1, 4], [2, 4], [3, 4], [4, 4], [5, 4],
        [6, 1], [10, 10], [11, 11], [5, 7], [7, 7]
    ])

    y = np.array([
        -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,
        1,  1,  1, 1, 1
    ])

    print("one vs all")
    model1 = SVMModel()
    model1.fit(X, y)
    preds1 = model1.predict(X)
    print(preds1)

    print("one vs all (2)")
    model2 = SVMModel(learning_rate=0.05, n_iters=300)
    model2.fit(X, y)
    preds2 = model2.predict(X)
    print(preds2)


if __name__ == "__main__":
    main()

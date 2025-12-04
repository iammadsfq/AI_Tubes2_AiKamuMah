import numpy as np
import pickle


class SVMModel:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, mode='one-vs-all'):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.mode = mode # 'one-vs-all', 'one-vs-one', 'dagsvm'
        self.models = [] # Menyimpan sub-model untuk multiclass

    def fit(self, X, y):
        classes = np.unique(y)

        if self.mode == 'one-vs-all':
            for c in classes:
                # Buat binary target: 1 jika kelas c, -1 jika bukan
                y_binary = np.where(y == c, 1, -1)
                model = SVM_Binary(self.lr, self.lambda_param, self.n_iters)
                model.fit(X, y_binary)
                self.models.append((c, model))

        elif self.mode == 'one-vs-one':
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    c1, c2 = classes[i], classes[j]
                    idx = np.where((y == c1) | (y == c2))[0]
                    X_ij = X[idx]
                    y_ij = np.where(y[idx] == c1, 1, -1)
                    model = SVM_Binary(self.lr, self.lambda_param, self.n_iters)
                    model.fit(X_ij, y_ij)
                    self.models.append(((c1, c2), model))

        elif self.mode == 'dagsvm':
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    c1, c2 = classes[i], classes[j]
                    idx = np.where((y == c1) | (y == c2))[0]
                    X_ij = X[idx]
                    y_ij = np.where(y[idx] == c1, 1, -1)
                    model = SVM_Binary(self.lr, self.lambda_param, self.n_iters)
                    model.fit(X_ij, y_ij)
                    self.models.append(((c1, c2), model))

    def predict(self, X):
        if self.mode == 'one-vs-all':
            scores = []
            for (c, model) in self.models:
                scores.append(model.decision_function(X))
            scores = np.array(scores)
            idx = np.argmax(scores, axis=0)
            return np.array([self.models[i][0] for i in idx])

        elif self.mode == 'one-vs-one':
            votes = {i: {} for i in range(X.shape[0])}
            for (c1, c2), model in self.models:
                pred = model.predict(X)
                for i, p in enumerate(pred):
                    if p == 1:
                        pick = c1
                    else:
                        pick = c2
                    votes[i][pick] = votes[i].get(pick, 0) + 1
            final = []
            for i in range(X.shape[0]):
                final.append(max(votes[i], key=votes[i].get))
            return np.array(final)

        elif self.mode == 'dagsvm':
            results = []
            for x in X:
                remaining = list(self.classes)
                while len(remaining) > 1:
                    c1 = remaining[0]
                    c2 = remaining[1]
                    for (p1, p2), model in self.models:
                        if (p1 == c1 and p2 == c2) or (p1 == c2 and p2 == c1):
                            pred = model.predict(x.reshape(1, -1))[0]
                            break
                    if pred == 1:
                        remain = c2
                    else:
                        remain = c1
                    remaining.remove(remain)
                results.append(remaining[0])
            return np.array(results)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class SVM_Binary:
    # Class helper untuk SVM biner (digunakan oleh multiclass wrapper)
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for j in range(n_samples):
                cond = y[j] * (np.dot(X[j], self.w) + self.b) >= 1
                if cond:
                    self.w -= self.lr * (self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (self.lambda_param * self.w - y[j] * X[j])
                    self.b += self.lr * y[j]

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def main():
    print()


if __name__ == "__main__":
    main()

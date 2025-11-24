import numpy as np
import pickle

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = [] # Untuk bonus video
        self.history_params = [] # Untuk bonus video

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # TODO: Implementasi forward pass (linear model + sigmoid)
            
            # TODO: Hitung Gradient dw dan db
            
            # TODO: Update weights dan bias
            
            # Simpan history untuk bonus visualisasi
            pass

    def predict(self, X):
        # TODO: Implementasi prediksi kelas (0 atau 1) berdasarkan threshold 0.5
        pass

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
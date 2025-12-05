import numpy as np
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin

class SVMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, mode='one-vs-all'):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.mode = mode # 'one-vs-all', 'one-vs-one', 'dagsvm'
        self.models = [] # Menyimpan sub-model untuk multiclass

    def fit(self, X, y):
        # TODO: Implementasi Multiclass Wrapper
        classes = np.unique(y)
        
        if self.mode == 'one-vs-all':
            for c in classes:
                # Buat binary target: 1 jika kelas c, -1 jika bukan
                y_binary = np.where(y == c, 1, -1)
                model = SVM_Binary(self.lr, self.lambda_param, self.n_iters)
                model.fit(X, y_binary)
                self.models.append((c, model))
        
        # TODO: Implementasi One-vs-One atau DAGSVM jika dipilih
        pass

    def predict(self, X):
        # TODO: Implementasi voting/agregasi prediksi dari sub-models
        pass
        
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
        # TODO: Implementasi training SVM (misal: Pegasos atau SMO sederhana)
        pass
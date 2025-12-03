import numpy as np
import numpy.typing as npt
import pickle
from typing import Literal, Optional

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

class LogisticRegressionModel:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000, multi_class: Literal['ovr', 'multinomial'] = 'ovr', random_state: Optional[int]= None):
        self.lr: float = learning_rate
        self.epochs: int = n_iters
        self.multi_class: Literal['ovr', 'multinomial'] = multi_class
        self.random_state: Optional[int] = random_state
        self.weights: Optional[FloatArray] = None
        self.bias = None
        self.classes = None
        self.losses = []
        self.history_params = []

    def _sigmoid(self, x: FloatArray) -> FloatArray:
        return 1 / (1 + np.exp(-x))
    
    def _softmax(self, x: FloatArray) -> FloatArray:
        # TODO: Implementasi fungsi softmax
        pass

    def fit(self, X: FloatArray, y: FloatArray):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        if self.multi_class == 'ovr':
            self._fit_ovr(X, y, n_samples, rng)
        else:
            self._fit_softmax(X, y, n_samples, n_features, n_classes)
    
    def _fit_ovr(self, X: FloatArray, y: FloatArray, n_samples: int, rng: np.random.Generator) -> None:
        for class_idx, class_label in enumerate(self.classes):
            y_classes = np.where(y == class_label, 1.0, 0.0)
            for _ in range(self.epochs):
                indices = rng.permutation(n_samples)
                for i in indices:
                    xi = X[i]
                    yi = y_classes[i]

                    wi = self.weights[:, class_idx]
                    bi = self.bias[class_idx]

                    z = np.dot(xi, wi) + bi

                    pi = float(self._sigmoid(z))

                    error = yi - pi

                    self.weights[:, class_idx] += (self.lr * error * xi)
                    self.bias[class_idx] += (self.lr * error)



    def _fit_softmax(self, X: FloatArray, y: FloatArray, n_samples: int, n_features: int, n_classes: int) -> None:
        # TODO: Implementasi Softmax Regression
        pass

    def predict(self, X: FloatArray):
        # TODO: Bikin predict untuk softmax regression
        z = np.dot(X, self.weights) + self.bias

        sigmoid_probability = self._sigmoid(z)

        class_indices = np.argmax(sigmoid_probability, axis=1)
        return self.classes[class_indices]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)